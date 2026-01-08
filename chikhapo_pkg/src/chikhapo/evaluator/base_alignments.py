from .base import BaseEvaluator

import os
import subprocess
import tempfile

class BaseAlignmentsEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.fastalign_dir = os.path.normpath(
            os.path.join(
                current_dir,
                "..", "..", "dep",
            )
        )
        self.fastalign_binary = os.path.normpath(
            os.path.join(
                self.fastalign_dir, "fast_align",
                "build", "fast_align"
            )
        )
        self.verify_fastalign_is_installed()

    def verify_fastalign_is_installed(self):
        if os.path.exists(self.fastalign_binary):
            return
        os.makedirs(self.fastalign_dir, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/clab/fast_align.git"],
            cwd=self.fastalign_dir,
            check=True,
        )
        self.verify_cmakelist()
        build_dir = os.path.join(self.fastalign_dir, "fast_align", "build")
        os.makedirs(build_dir, exist_ok=True)
        subprocess.run(
            ["cmake", ".."],
            cwd=build_dir,
            check=True
        )
        subprocess.run(
            ["make"],
            cwd=build_dir,
            check=True
        )

    def verify_cmakelist(self):
        cmakelist_file_path = os.path.join(self.fastalign_dir, "fast_align", "CMakeLists.txt")
        if not os.path.exists(cmakelist_file_path):
            raise Exception("The file needed to run make CMakeLists.txt is missing")
        with open(cmakelist_file_path, "r") as f:
            lines = f.readlines()
        correct_version = "cmake_minimum_required(VERSION 3.5)\n"
        if lines[1] != correct_version:
            lines[1] = correct_version
        with open(cmakelist_file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        
        # build_dir = os.path.join(self.fastalign_dir, "build")
        # os.makedirs(build_dir, exist_ok=True)
        # subprocess.run(
        #     ["make"],
        #     cwd=build_dir,
        #     check=True
        # )

    def convert_src_tgt_sentences_to_temp_file(self, reverse=False):
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", prefix="fastalign_input_")
        for entry in self.data:
            # When will we use alignments: X->eng direction. In other words, 
            # we will use alignments in the case where we have an English word 
            # and want to translate into the (source) language X. Therefore we 
            # have to reverse the direction of translation
            if reverse:
                src_sentence, tgt_sentence = entry["tgt_sentence_gt"], entry["src_sentence"]
            else:
                src_sentence, tgt_sentence = entry["src_sentence"], entry["tgt_sentence_gt"]
            temp_file.write(f"{src_sentence} ||| {tgt_sentence}\n")
        temp_file.close()
        return temp_file.name
    
    def run_fastalign(self, input_file):
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".align", prefix="fastalign_output_").name
        cmd = [self.fastalign_binary, "-i", input_file, "-v", "-o", "-d"]
        with open(output_file, "w") as out_f:
            subprocess.run(
                cmd,
                stdout=out_f,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
        return output_file
            
    def process_fastalign_alignments(self, int_to_int_alignments, reverse=False):
        srcWord_to_tgtWord_alignments = {}        
        for i, (entry, int_to_int_alignment) in enumerate(zip(self.data, int_to_int_alignments)):
            if reverse:
                src_sentence, tgt_sentence = entry["tgt_sentence_gt"], entry["src_sentence"]
            else:
                src_sentence, tgt_sentence = entry["src_sentence"], entry["tgt_sentence_gt"]
            src_words = src_sentence.split()
            tgt_words = tgt_sentence.split()
            int_to_int_list = int_to_int_alignment.split()
            
            if i not in srcWord_to_tgtWord_alignments:
                srcWord_to_tgtWord_alignments[i] = {}
            
            for int_to_int in int_to_int_list:
                if len(int_to_int.split("-")) != 2:
                    raise Exception(f"The alignment {int_to_int} should be demarcated with a - and should not include negatives.")
                src_int, tgt_int = tuple(map(int, int_to_int.split("-")))
                if src_int >= len(src_words):
                    raise Exception(f"Source alignment {int_to_int} should be within the number of words in the source sentence.")
                if tgt_int >= len(tgt_words):
                    raise Exception(f"Target alignment {int_to_int} should be within the number of words in the target sentence.")
                src_word, tgt_word = src_words[src_int], tgt_words[tgt_int]
                srcWord_to_tgtWord_alignments[i].setdefault(src_word, {})
                srcWord_to_tgtWord_alignments[i][src_word][tgt_word] = \
                    srcWord_to_tgtWord_alignments[i][src_word].get(tgt_word, 0) + 1
        
        return srcWord_to_tgtWord_alignments

    def get_statistical_alignments(self, reverse=False):
        input_file = None
        alignments_file = None
        try:
            input_file = self.convert_src_tgt_sentences_to_temp_file(reverse=reverse)
            alignments_file = self.run_fastalign(input_file)
            with open(alignments_file, "r") as f:
                int_to_int_alignments = [line.strip() for line in f]
            word_to_word_alignments = self.process_fastalign_alignments(int_to_int_alignments, reverse=reverse)
            return word_to_word_alignments
        finally:
            if input_file and os.path.exists(input_file):
                os.unlink(input_file)
            if alignments_file and os.path.exists(alignments_file):
                os.unlink(alignments_file)
                