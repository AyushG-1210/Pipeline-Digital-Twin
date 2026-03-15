path = r"C:\Users\anshu\Desktop\Major Project\Audited_Physics_Index.csv"

import pandas as pd

df = pd.read_csv(path)
df.drop_duplicates(inplace=True)
df.drop(df[df["Physics_Score"] < 5].index, inplace=True)
df.drop_duplicates(subset=["Original_Header"], inplace=True)
df.to_csv(r"C:\Users\anshu\Desktop\Major Project\Cleaned_Physics_Index.csv", index=False)