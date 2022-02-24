import json
import logging
import nltk
import os.path
import sumy

from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.client_config import ClientConfig
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.io.file_ref import FileRef
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation

from newspaper import Article
from os.path import join

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.kl import KLSummarizer

from zipfile import ZipFile
nltk.download('punkt')



def fetch_article(url):

    '''
    Function for fetching an article from a web page with newspaper3k
    Input:  url (str)  --> article URL
    Output: text (str) --> article
    '''
    
    article = Article(url)
    article.download()
    article.parse()
    text = article.text
    
    text = text.strip().replace('\n',' ')
    
    return text


def preprocess_data(infile, base_path='./'):
    
    '''
    Prepocess text data for use in summarization
    Input:  infile (str)        --> name of the input file (currently only PDF and .txt files are supported)
            base_path (str)     --> full path of the base directory for the input and output files
    Output: fulltext (str)      --> extracted text
            data (dict)         --> full extracted PDF content (only of infile is PDF)
            headers (dict)      --> dictionary with names of section headers (only if infile is PDF)
            headers_list (list) --> list of dictionaries with names of section headers and their positions in fulltext
            
            N.B.: A call to this funtion creates a .zip-file in base_path, if the input file is a PDF. 
                  This file contains the extracted PDF content in JSON format.  
    '''
    
    data = {}
    fulltext = ''
    headers = {}
    headers_list = []
    file_type = infile[-3:]
    
    # Check for file type:
    if file_type == 'pdf':
        
        outfile = infile[:15] + '.zip'
        # Check if output file exists 
        if os.path.exists(join(base_path, outfile)):
            os.remove(join(base_path, outfile))
            
        # Extract content from PDF 
        # We use Adobe's Document Services API which is, obviously, the best for this task...
        # ...but which is also, even more obviously, not free of charge :( (no, PyPDF2 isn't good enough for this job)
        xtrct_txt_frm_pdf(infile, outfile, base_path=base_path)
        
        # Load the extracted PDF content  
        status, data = load_pdf_xtrct(filename=outfile)
        if status:
            for item in data['elements']:
                try:
                    # Concatenate all text elements 
                    fulltext += item['Text']
                except:
                    pass
            # Adobe's extract API replaces the links it found in a PDF with the string '(<>)'--let's get rid of it!
            fulltext = fulltext.replace('(<>)','')
            
            # Identify section headers
            for item in data['elements']:
                try:
                    path = item['Path']
                    head = item['Text']
                    if 'H1' in path and not 'Sub' in path:
                        headers[path] = head
                except:
                    pass
    
            # Locate start indices of section headers in fulltext
            for p,h in headers.items():
                idx = fulltext.find(h)
                if idx != -1:
                    headers_list.append({'path': p, 'header': h, 'idx_start': idx})

            # Add end indices of section headers 
            num_headers = len(headers_list)
            for i in range(num_headers-1):
                headers_list[i]['idx_end'] = headers_list[i+1]['idx_start']
            headers_list[-1]['idx_end'] = len(fulltext)

    elif file_type == 'txt':
        with open(base_path + infile, 'r', encoding='utf-8') as f:
            fulltext = f.read()
        fulltext = fulltext.strip().replace('\n',' ')
        
    else:
        print('Sorry, your file type is not supported at this time!')
        
    return fulltext, data, headers, headers_list

              
def xtrct_txt_frm_pdf(infile, outfile, base_path='./', log=False):
    
    '''
    Function to extract text content from a PDF file using the Adobe Document Services API
    Input: infile (str) --> name of the input PDF file
           outfile (str) --> name of the output zip-file
           log (boolean) --> switch logging on/off        
    
    The following code was copied from this source:
    https://github.com/adobe/pdfservices-python-sdk-samples
    Original Copyright statement:
        Copyright 2021 Adobe. All rights reserved.
        This file is licensed to you under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License. You may obtain a copy
        of the License at http://www.apache.org/licenses/LICENSE-2.0
    '''
    
    if log:
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    try:
        # Initial setup, create credentials instance.
        credentials = Credentials.service_account_credentials_builder() \
            .from_file(base_path + "pdfservices-api-credentials.json") \
            .build()

        # Create client config instance with custom time-outs.
        client_config = ClientConfig.builder().with_connect_timeout(40000).with_read_timeout(40000).build()

        # Create an ExecutionContext using credentials and create a new operation instance.
        execution_context = ExecutionContext.create(credentials, client_config)
        extract_pdf_operation = ExtractPDFOperation.create_new()

        # Set operation input from a source file.
        source = FileRef.create_from_local_file(join(base_path,infile))
        extract_pdf_operation.set_input(source)

        # Build ExtractPDF options and set them into the operation
        extract_pdf_options: ExtractPDFOptions = ExtractPDFOptions.builder() \
            .with_element_to_extract(ExtractElementType.TEXT) \
            .build()
        extract_pdf_operation.set_options(extract_pdf_options)

        # Execute the operation.
        result: FileRef = extract_pdf_operation.execute(execution_context)

        # Save the result to the specified location.
        result.save_as(join(base_path,outfile))

    except (ServiceApiException, ServiceUsageException, SdkException):
        logging.exception("Exception encountered while executing operation")


def load_pdf_xtrct(base_path='./', filename='PDFXtrctOut.zip'):
    
    '''
    Function to read output zip-file from xtrct_txt_frm_pdf
    Input:  base_path (str)  --> name of the directory for the output zip-file
            filename (str)   --> name of the output zip-file
    Output: status (boolean) --> reading succeeful (yes/no)
            data (list)      --> PDF data
    '''
    
    status = False
    data = None

    try:
        with ZipFile(join(base_path,filename), 'r') as zip:
            data = json.loads(zip.read("structuredData.json"))
        status = True
    except:
        print('ERROR: The specified file does not exist: ', join(base_path,filename))

    return status, data
              
           
def extractive_summarizers(text, algorithm='LSA', max_len=3, lang='english'):
    
    '''
    Create an extractive summary of an input text using different algorithms
    Input:  text (str)      --> text to summarize
            method (str)    --> algorithm for extractive summarization: LRS, LSA, LUHN. or KLS
            max_len (int)   --> length of the summary (=number of sentences)
            lang (str)      --> language of the input text and the summary
    Output: summ_out (str)  --> summary of text in max_len sentences
    '''
    
    method = algorithm.lower()
    
    parsed = PlaintextParser.from_string(text, Tokenizer(lang))
    summary = ()

    if method=='lrs':
        lex_rank_summarizer = LexRankSummarizer()
        summary = lex_rank_summarizer(parsed.document, sentences_count=max_len)
    elif method=='lsa':
        lsa_summarizer = LsaSummarizer()
        summary = lsa_summarizer(parsed.document, sentences_count=max_len)
    elif method=='luhn':
        luhn_summarizer = LuhnSummarizer()
        summary = luhn_summarizer(parsed.document, sentences_count=max_len)
    elif method == 'kls':
        kl_summarizer = KLSummarizer()
        summary = kl_summarizer(parsed.document, sentences_count=max_len)
    
    summ_out = ''
    for sentence in summary:
        summ_out += str(sentence) + ' '
    
    return summ_out


def gpt2_summarizer(tokenizer, generator, text, max_sum_length=128, temperature=0.7):
    
    '''
    Function to generate an abstractive summary of an input text using Huggingface's GPT-2 implementation
    Input:  tokenizer            --> GPT-2 tokenizer object
            generator            --> GPT-2 model instance
            text (str)           --> text to summarize
            max_sum_length (int) --> maximum number of tokens for the generated summary.
            temperature (float)  --> GPT-2 parameter for controlling output randomness
    Output: sum_trunc (str)      --> summary of text (unfinished sentences truncated)
    '''
    
    # Tell GPT-2 what to do
    prompt = f"Text: {text}\n\n\Summary:"
    
    n_tokens = len(tokenizer(prompt)['input_ids'])
    max_tokens = n_tokens + max_sum_length 
    summaries = generator(prompt,
                        max_length=max_tokens,
                        temperature=temperature,
                        top_p=1.0,
                        return_full_text=False,
                        num_return_sequences=1)

    summary = summaries[0]['generated_text']
    
    # Truncate the last unfinished sentence (if there is one)--not optimal for a summary, but has to do for now 
    sum_trunc = ''
    periods = [pos for pos, char in enumerate(summary) if char == '.']
    if len(periods) != 0:
        cut_pos = periods[-1] + 1 
        sum_trunc = summary[:cut_pos]
    else:
        sum_trunc = summary
    
    return sum_trunc
    

def chunk_text(text, max_tokens=400):
    
    '''
    Function for spliting text into equal chunks of length less than max_tokens 
    Input:  text (str)       --> text to split into chunks
            max_tokens (int) --> maximum number of tokens for each chunk
    Output: chunks (list)    --> list of text chunks
    '''
    
    chunks = []
    chunk = ''
    num_tokens = len(nltk.word_tokenize(text))
  
    count_tokens = 0
    count_chunks = 0
    if num_tokens > max_tokens:
        num_chunks = num_tokens//max_tokens
        chunk_size = num_tokens//(num_chunks + 1)
        print(f"INFO: Text was split into {num_chunks + 1} chunks of <={chunk_size} tokens")  
        for sentence in nltk.sent_tokenize(text):
            chunk += sentence + ' '
            count_tokens = len(nltk.word_tokenize(chunk))
            if count_tokens >= chunk_size and count_chunks < num_chunks:
                chunks.append(chunk.strip())
                count_chunks += 1
                chunk = ''
            elif count_chunks == num_chunks:
                continue
        chunks.append(chunk.strip())
    else:
        chunks.append(text)
        print(f"INFO: No need to chunk--text is short enough for transformer ({num_tokens}<{max_tokens})")
        
    return chunks