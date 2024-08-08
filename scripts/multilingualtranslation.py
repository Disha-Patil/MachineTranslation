import os
from collections import OrderedDict
import pandas as pd
import re
import spacy
from spacy.lang.ko.examples import sentences 
import docx
import pymupdf
from llama_cpp import Llama




class MultiLangTranslation():
    """
    initiates an instance with source and target languages for translation.
    Args:
        source_lang : language to be translated
        target_lang : translation language
        nlp_obj : spacy language model for source language. default is korean, or you can specify anyother
                  by calling nlp_obj = spacy.load("language_model")
                  refer - https://spacy.io/usage/models
                
    """
    def __init__(self, source_lang, target_lang, model_name_or_path,model_file, n_ctx=None,nlp_obj=None):
        self.src=source_lang
        self.trg=target_lang
        self.nlp_obj = spacy.load("ko_core_news_lg") if nlp_obj is None else nlp_obj

        self.model_name_or_path =model_name_or_path
        self.model_file =model_file
        user_n_ctx=512 if n_ctx is None else n_ctx
        self.llm = Llama.from_pretrained(
                    repo_id=model_name_or_path,
                    filename=model_file,#*q8_0.gguf",
                    verbose=False,
                    n_ctx=user_n_ctx
                )
        print(f"n_ctx is set to {self.llm.n_ctx()}")

    def getText_docx(self,filename:str,)->str:
        doc = docx.Document(filename)
        fullText = [' ' + para.text for para in doc.paragraphs]
        return '\n\n'.join(fullText)
    
    def getText_pdf(self,filename:str,)->str:
        doc = pymupdf.open(filename)
        fullText=[]
        for page in doc: # iterate the document pages
            text = page.get_text().encode("utf8") # get plain text (is in UTF-8)
            fullText.append(' ' + text)
        return '\n\n'.join(fullText)
    
    def getText_txt(self,filename:str,)->str:
        with open(filename, 'r') as doc:
            fullText = doc.read()
        return fullText

    def readText(self,data_dir:str)->dict:
        """
        takes in directory path where files to be translated are stores.
        reads files. 
        converts text into sentences.
        return dict with filenames and text

        Args : 
            data_dir : path of directory where data is stored in readable files. 
                       Make sure it has only data files that you wish to read.
                       Make sure the filenames are unique.
            file_type: accepted file types are - docx, txt, pdf
        Returns :
            A dict with filename as keys and the text extracted from those files as values.
        """
        file_list=[data_dir+file_name for file_name in os.listdir(data_dir)]
        ExtractedText={}
        for filename in file_list:      
            if filename.endswith("docx"):
                path_head, path_tail = os.path.split(filename)
                ExtractedText[path_tail]=self.getText_docx(filename)
            
            elif filename.endswith("pdf"):
                path_head, path_tail = os.path.split(filename)
                ExtractedText[path_tail]=self.getText_pdf(filename)

            elif filename.endswith("txt"):
                path_head, path_tail = os.path.split(filename)
                ExtractedText[path_tail]=self.getText_txt(filename)

            else:
                raise Exception("Filetype not supported, please check if your file is docx,pdf or txt.")


        return ExtractedText
    
    
    def get_sentences(self,txt_doc):
        """
        takes in raw text and converts it to sentences.
        """
        txt_doc=self.nlp_obj(txt_doc)
        sentences=[sent.text.strip() for sent in txt_doc.sents]
        return sentences 
    
    def get_paragraphs(self,txt_doc,para_delimited):
        """
        if you know your paragraph delimiter you can use this function. 
        TODO:
        use a model for identifying paragraph chunks. 
        """
        return txt_doc.str.split(para_delimited)

    def get_translation(self,prompt):
        try:
            llm_output=self.llm.create_chat_completion(
                messages = [
                    {"role": "system", 
                     "content": f"Translate the following text from {self.src} to {self.trg}. Give only the translation and no meaning."},
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return llm_output['choices'][0]['message']['content']
        except Exception as err:
            print(err)

    def get_translation_rev(self,prompt):

        try:

            llm_output=self.llm.create_chat_completion(
                messages = [
                    {"role": "system", "content": f"Translate the following text from {self.trg} to {self.src}. Give only the translation and no meaning."},
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return llm_output['choices'][0]['message']['content']
        except Exception as err:
                print(err)
