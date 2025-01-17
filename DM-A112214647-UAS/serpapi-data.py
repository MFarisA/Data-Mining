import serpapi
import os
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('SERPAPI_KEY')
client = serpapi.Client(api_key=api_key)

results = client.search(
    engine="google_play_product",
    product_id="com.miHoYo.GenshinImpact",
    store="apps",
    all_reviews="true",
    num=199
)

data = results['reviews']
print("total reviews : ", len(results['reviews']))

print("all done")
df = pd.DataFrame(data)
df.to_csv('google-play-rev-gen-2.csv', index=False)