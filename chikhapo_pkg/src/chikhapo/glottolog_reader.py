import os
import pandas as pd
from pathlib import Path
import pycountry
import urllib.request
import zipfile

class GlottologReader:
    """
    Glottolog reader retrieves information for individual languages as well as languages 
    within a family.
    """
    def __init__(self):
        cache_dir = Path.home() / '.cache' / 'chikhapo'
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.glottolog_path = os.path.normpath(
            os.path.join(
                cache_dir,
                "glottolog",
                "languoid.csv",
            )
        )
        self.glottolog_df = None

    def verify_glottolog_is_installed(self):
        GLOTTOLOG_URL = (
            "https://cdstar.eva.mpg.de//bitstreams/EAEA0-2198-D710-AA36-0/glottolog_languoid.csv.zip"
        )
        if os.path.exists(self.glottolog_path):
            return
        parent_dir = os.path.dirname(self.glottolog_path)
        os.makedirs(parent_dir, exist_ok=True)
        zip_path = os.path.join(parent_dir, "languoid.csv.zip")
        urllib.request.urlretrieve(GLOTTOLOG_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract("languoid.csv", parent_dir)
        os.remove(zip_path)

    def get_lang_info(self, iso):
        if self.glottolog_df is None:
            self.verify_glottolog_is_installed()
            self.glottolog_df = pd.read_csv(self.glottolog_path)
        if len(iso) != 3:
            raise Exception("Please enter a valid ISO code")
        if iso not in self.glottolog_df["iso639P3code"].values:
            raise Exception(f"The iso {iso} could not be found in the Glottolog data.")
        iso_df = self.glottolog_df.loc[self.glottolog_df["iso639P3code"] == iso]
        info = []
        for _, row in iso_df.iterrows():
            country_ids = row["country_ids"].split()
            countries = []
            for country_id in country_ids:
                country_name = pycountry.countries.get(alpha_2=country_id).name
                countries.append(country_name)
            
            info.append({
                "name": row["name"],
                "iso": iso,
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "country": countries
            })
        return info

    def get_language_to_family_dict(self):
        if self.glottolog_df is None:
            self.verify_glottolog_is_installed()
            self.glottolog_df = pd.read_csv(self.glottolog_path)
        language_to_family = {}
        glottolog_languages_df = self.glottolog_df.loc[self.glottolog_df["level"]=="language"]
        glottolog_languages_df = glottolog_languages_df[["family_id", "iso639P3code"]]
        glottolog_families_df = self.glottolog_df[self.glottolog_df["level"]=="family"]
        glottolog_families_df = glottolog_families_df[["id", "name"]]
        glottolog_languages_and_families_df = pd.merge(glottolog_languages_df, glottolog_families_df, 
                                                       left_on="family_id", right_on="id", how="inner")
        glottolog_languages_and_families_df = glottolog_languages_and_families_df.dropna()
        for _, row in glottolog_languages_and_families_df.iterrows():
            lang = row["iso639P3code"]
            fam = row["name"]
            language_to_family[lang] = fam
        return language_to_family
    