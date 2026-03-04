import zipfile
import csv
import os
import re
import pandas as pd
from tqdm import tqdm
from utils import safe_save2file
HOME_DIR = os.path.expanduser("~/HumanLLM_data")

ZIP_PATH = f"{HOME_DIR}/raw/blogs.zip"
OUTPUT_CSV = f"{HOME_DIR}/blogger/blogtext_v1.csv"


def parse_filename(filename):
    """
    7596.male.26.Internet.Scorpio.xml
    """
    name = os.path.basename(filename)
    user_id, gender, age, topic, sign, _ = name.split(".")
    return user_id, gender, age, topic, sign


def main():

    with zipfile.ZipFile(ZIP_PATH, "r") as z:

        xml_files = [f for f in z.namelist() if f.endswith(".xml")]

        print("Total users:", len(xml_files))

        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:

            writer = csv.writer(fout)
            writer.writerow(["id", "gender", "age", "topic", "sign", "date", "text"])

            total_posts = 0

            for file in xml_files:

                user_id, gender, age, topic, sign = parse_filename(file)

                content = z.read(file).decode("latin_1")

                dates = re.findall(r"<date>(.*?)</date>", content)
                posts = re.findall(r"<post>(.*?)</post>", content, re.DOTALL)

                for date, post in zip(dates, posts):

                    text = post.replace("\n", " ").strip()

                    writer.writerow([
                        user_id,
                        gender,
                        age,
                        topic,
                        sign,
                        date,
                        text
                    ])

                    total_posts += 1

    print("Total posts:", total_posts)
    print("CSV saved to:", OUTPUT_CSV)

def filter_dataset():
    infile = OUTPUT_CSV
    outfile = f"{HOME_DIR}/blogger/blogtext_v2.csv"
    
    # create folder recursively if not exists 
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    
    try:
        df = pd.read_csv(infile, header=0)
    except Exception as e:
        print(f"file {infile} is not a valid csv file, skip")
        return
    ## print the length of df 
    print(len(df))
    
    ## remove rows that the text column length is less that 300 
    df = df[df["text"].apply(lambda x: len(x) > 300)]
    print(len(df))
    safe_save2file(outfile, df)

def split_by_ids():
    infile = f"{HOME_DIR}/blogger/blogtext_v2.csv"
    outdir = f"{HOME_DIR}/blogger/by_users"
    os.makedirs(outdir, exist_ok=True)
    
    try:
        df = pd.read_csv(infile, header=0)
    except Exception as e:
        print(f"file {infile} is not a valid csv file, skip")
        return
    total_cnt = 0 
    error_cnt = 0
    valid_cnt = 0
    user_cnt, user_error = 0, 0
    
    ## split the data by the id column, output each id's data to a separate file under the outdir, order by date column     
    for name, group in tqdm(df.groupby("id"), total=len(df["id"].unique()), desc="Splitting"): #  df.groupby("id"):
        ## remove rows that the date column is empty or length is less than 5
        user_cnt += 1
        all_rows_cnt = len(group) 
        group = group[group["date"].apply(lambda x: len(str(x)) > 5)]
        clean_rows_cnt = len(group)
        
        total_cnt += all_rows_cnt 
        error_cnt += all_rows_cnt - clean_rows_cnt
        
        if len(group) == 0:
            continue
        
        try:
            ## the date column is date but in string format like "05,April,2001", so need to convert to datetime format before sorting in ascending order
            group["date"] = pd.to_datetime(group["date"], format="%d,%B,%Y")
        except:
            user_error += 1
            continue
        
        group = group[group["text"].apply(lambda x: len(str(x)) > 100)]  
        if len(group) == 0:
            continue
        
        group = group.sort_values(by="date", ascending=True) 
        group.to_csv(os.path.join(outdir, str(name) + ".csv"), index=False)
        valid_cnt += len(group)
    
    print("total_cnt: ", total_cnt, "error_cnt: ", error_cnt, "valid_cnt: ", valid_cnt)
    print("user_cnt: ", user_cnt, "user_error: ", user_error)


if __name__ == "__main__":
    main()
    filter_dataset()
    split_by_ids()