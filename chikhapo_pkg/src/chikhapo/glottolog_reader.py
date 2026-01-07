import pandas as pd
import pycountry
import os

class GlottologReader:
    def __init__(self):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        glottolog_path = os.path.join(
            current_file_dir, 
            "..", "..", "..",  # Changed from 4 to 3
            "glottolog_languoid.csv", 
            "languoid.csv"
        )
        self.glottolog_path = os.path.normpath(glottolog_path)
        self.glottolog_df = pd.read_csv(self.glottolog_path)

    def get_lang_info(self, iso):
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
        if not os.path.exists(self.glottolog_path):
            raise Exception("The path glottolog_languoid.csv/languoid.csv does not exist. Either, \n"\
                            "(i) go to the Glottolog downloads page https://glottolog.org/meta/downloads to download the most recent version OR\n"\
                            "(ii) verify that the file is placed in the correct place. Please use set_glottolog_path(...) if nexessary.")
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