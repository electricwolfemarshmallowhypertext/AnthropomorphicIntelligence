## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

import time
import sys
import torch
import argparse
from torch.cuda.amp import autocast
from model.model import PCC
from datasets import load_dataset



def run(args):
    data_path = "BroAlanTaps/efficiency_samples_8k"
    ds = load_dataset(data_path)['train']
    device = 'cuda:0'

    ratio = args.ratio
    batch_size = args.batch_size
    num_batches = 0  

    input_length = args.input_length
    generate_length = args.generate_length

    normal_prefilling_total_time = 0
    nornal_decode_total_time = 0

    comp_prefilling_total_time = 0
    comp_decode_total_time = 0
    compress_total_time = 0 
    
    args = argparse.Namespace(
        device=device,
        compress_model='BroAlanTaps/Stage1-PCC-Lite-4x',
        converter_model='BroAlanTaps/Stage1-PCC-Lite-4x',
        decoder_model='meta-llama/Meta-Llama-3-8B-Instruct',
        stage=2,
        segment_length=256,
        embed_len=256 // ratio,
        drop_out=0,
        use_lora=False,
        compressor_gradient_checkpoint=False,
        decoder_gradient_checkpoint=False
    )
    print(args)
    model = PCC(args).to(device).eval()

    tokenizer = model.compressor.tokenizer



    for idx, i in enumerate(range(0, len(ds), batch_size)):
        if idx >= 5:
            break
        
        # Collect data
        data = ds[i:i + batch_size]
        batch_texts = [item for item in data['text']]
        compress_ids = tokenizer(batch_texts, max_length=input_length, truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
        llm_ids = model.decoder.tokenizer(batch_texts, max_length=input_length, truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
        print(compress_ids.shape)
        print("-" * 25 + "Begin Test" + "-" * 25)
        # break
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Normal Mode(Only LLM):
                # Normal Prefilling
                past_key_values = None
                torch.cuda.synchronize()
                normal_prefilling_begin_time = time.time()
                llm_embedding = model.decoder.model.get_input_embeddings()(llm_ids)
                outputs = model.decoder.model(inputs_embeds=llm_embedding,past_key_values=past_key_values,use_cache=True)
                past_key_values = outputs.past_key_values
                torch.cuda.synchronize()
                normal_prefilling_end_time = time.time()
                print(f"Normal Prefilling time: {(normal_prefilling_end_time - normal_prefilling_begin_time) * 1000:.3f} ms")

                # Normal Decoding
                torch.cuda.synchronize()
                start_normal_decode_time = time.time()
                next_token_logits = outputs.logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
                for _ in range(1, generate_length):
                    current_token_embeds = model.decoder.model.get_input_embeddings()(next_tokens.unsqueeze(1))
                    with autocast(dtype=torch.bfloat16):
                        outputs = model.decoder.model(
                            inputs_embeds=current_token_embeds,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                
                # torch.cuda.synchronize()
                end_normal_decode_time = time.time()

                # Compressed Mode(With Compression):
                torch.cuda.synchronize()
                start_comp_time = time.time()
                compress_embedding = model(compress_ids=compress_ids, llm_ids=None, get_embedding=True).to(device)
                torch.cuda.synchronize()
                end_comp_time = time.time()            

                # Compression Prefilling:
                pcc_past_key_values = None
                torch.cuda.synchronize()
                com_prefilling_begin_time = time.time()
                print(f"Compression Prefilling begin, input shape: {compress_embedding.shape}")
                pcc_outputs = model.decoder.model(inputs_embeds=compress_embedding,past_key_values=pcc_past_key_values,use_cache=True)
                pcc_past_key_values = pcc_outputs.past_key_values
                torch.cuda.synchronize()
                comp_prefilling_end_time = time.time()
                print(f"Comp Prefilling time: {(comp_prefilling_end_time - com_prefilling_begin_time) * 1000:.3f} ms")
                
                
                # Compression Decoding:
                torch.cuda.synchronize()
                start_com_decode_time = time.time()
                next_token_logits = pcc_outputs.logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1)

                for _ in range(1, generate_length):
                    current_token_embeds = model.decoder.model.get_input_embeddings()(next_tokens.unsqueeze(1))
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pcc_outputs = model.decoder.model(
                            inputs_embeds=current_token_embeds,
                            past_key_values=pcc_past_key_values,
                            use_cache=True
                        )
                    pcc_past_key_values = pcc_outputs.past_key_values
                    next_token_logits = pcc_outputs.logits[:, -1, :]
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                torch.cuda.synchronize()
                end_com_decode_time = time.time()
    
                
                
        if idx == 0:
            continue
        
        compress_batch_time = end_comp_time - start_comp_time
        compress_total_time += compress_batch_time
        
        comp_prefilling_batch_time = comp_prefilling_end_time - com_prefilling_begin_time
        comp_prefilling_total_time += comp_prefilling_batch_time
        
        comp_decode_batch_time = end_com_decode_time - start_com_decode_time
        comp_decode_total_time += comp_decode_batch_time
        
        normal_prefilling_batch_time = normal_prefilling_end_time - normal_prefilling_begin_time
        normal_prefilling_total_time += normal_prefilling_batch_time
        
        normal_decode_batch_time = end_normal_decode_time - start_normal_decode_time
        nornal_decode_total_time += normal_decode_batch_time
        
        num_batches += 1
        
        
    compress_average_time = compress_total_time / num_batches if num_batches > 0 else 0
    compress_prefilling_average_time = comp_prefilling_total_time / num_batches if num_batches > 0 else 0
    compress_decode_average_time = comp_decode_total_time / num_batches if num_batches > 0 else 0

    normal_prefilling_average_time = normal_prefilling_total_time / num_batches if num_batches > 0 else 0
    normal_decode_average_time = nornal_decode_total_time / num_batches if num_batches > 0 else 0


    log_file = open(f"./experience/efficiency/input{input_length}_generate{generate_length}_batch{batch_size}.log", "w")
    original_stdout = sys.stdout
    sys.stdout = log_file

    print(f"-"*20 + "Final Result" + "-"*20)
    print(f"Setting: \ninput_length {input_length} \ngenerate_length {generate_length} \nbatch_size {batch_size} \nratio {ratio}")

    print(f"Average compression time: {compress_average_time * 1000:.2f} ms")
    print(f"Average comp. prefilling time : {compress_prefilling_average_time * 1000:.2f} ms")
    print(f"Average comp. decoding time: {compress_decode_average_time * 1000:.2f} ms")

    print(f"Average normal prefilling time: {normal_prefilling_average_time * 1000:.2f} ms")
    print(f"Average normal decoding time: {normal_decode_average_time * 1000:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PCC Efficiency")
    parser.add_argument("--ratio", type=int, default=4, help="Compression ratio")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--input_length", type=int, default=1024, help="Input length for the model")
    parser.add_argument("--generate_length", type=int, default=32, help="Length of the generated sequence")
    args = parser.parse_args()
    run(args)