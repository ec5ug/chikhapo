from .base import BaseTaskFeeder

class FillInTheBlankFeeder(BaseTaskFeeder):
    def get_lang_pairs(self, DIRECTION=None):
        flores_subset_names = self.loader.get_flores_subset_names()
        flores_subset_names.remove("eng_Latn")
        to_eng = []
        from_eng = []
        for name in flores_subset_names:
            to_eng.append(f"{name}_eng")
            from_eng.append(f"eng_{name}")
        if DIRECTION is None:
            return to_eng+from_eng
        elif DIRECTION=="X_to_eng":
            return to_eng
        elif DIRECTION=="eng_to_X":
            return from_eng
        else:
            raise Exception("An invalid directon was specified. It should be None, \"X_to_eng\", or \"eng_to_X\"")
        