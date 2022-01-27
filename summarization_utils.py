import json
import logging
import nltk
import sumy
import os.path

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
    
    text = text.replace('\n\n',' ')
    
    return text


def preprocess_data(infile, base_path='./'):
    
    '''
    Prepocess text data for use in summarization
    Input:  infile (str)        --> name of the input file (currently only PDF and .txt files are supported)
            base_path (str)     --> full path of the base directory for the input and output files
    Output: fulltext (str)      --> extracted text
            data (dict)         --> full extracted PDF content (only of infile is PDF)
            headers (dict)      --> dictionary with names of section headers (only if infile is PDF)
            headers_list (list) --> list of dictionaries with names of section headers and their locations in fulltext
            
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
        fulltext = fulltext.replace('\n\n',' ')
        
    else:
        print('Sorry, your file type is not supported at this time!')
        
    return fulltext, data, headers, headers_list

              
def xtrct_txt_frm_pdf(infile, outfile, base_path='./', log=False):
    
    '''
    Function to extract text content from a PDF file using the Adobe Document Services API
    Input: infile (str) --> name of the input PDF file
           outfile (str) --> name of the output zip-file
           log (boolean) --> switch logging on/off
    '''
    
    if log:
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    try:
        # Initial setup, create credentials instance.
        credentials = Credentials.service_account_credentials_builder() \
            .from_file(base_path + "pdfservices-api-credentials.json") \
            .build()

        # Create client config instance with custom time-outs.
        client_config = ClientConfig.builder().with_connect_timeout(10000).with_read_timeout(40000).build()

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
              
           
def extractive_summarizers(text, method='LRS', max_len=3, lang='english'):
    
    '''
    Create an extractive summary of an input text using different algorithms
    Input:  text (str)      --> text to summarize
            method (str)    --> algorithm for extractive summarization: LRS, LSA, LUHN. or KLS
            max_len (int)   --> length of the summary (=number of sentences)
            lang (str)      --> language of the input text and the summary
    Output: summary (tuple) --> summary of text in max_len sentences
    '''
    
    parsed = PlaintextParser.from_string(text, Tokenizer(lang))
    summary = ()

    if method=='LRS':
        lex_rank_summarizer = LexRankSummarizer()
        summary = lex_rank_summarizer(parsed.document, sentences_count=max_len)
    elif method=='LSA':
        lsa_summarizer = LsaSummarizer()
        summary = lsa_summarizer(parsed.document, sentences_count=max_len)
    elif method=='LUHN':
        luhn_summarizer = LuhnSummarizer()
        summary = luhn_summarizer(parsed.document, sentences_count=max_len)
    elif method == 'KLS':
        kl_summarizer = KLSummarizer()
        summary = kl_summarizer(parsed.document, sentences_count=max_len)

    return summary


def chunk_text(text, max_tokens=475):
    
    '''
    Function for spliting text into chunks of length less than max_token 
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
        for sentence in nltk.sent_tokenize(text):
            chunk += sentence
            count_tokens = len(nltk.word_tokenize(chunk))
            if count_tokens >= max_tokens and count_chunks < num_chunks:
                chunks.append(chunk)
                count_chunks += 1
                chunk = ''
            elif count_chunks == num_chunks:
                continue
        chunks.append(chunk)
    else:
        chunks.append(text)
        print(f"INFO: No need to chunk--text is short enough for transformer ({num_tokens}<{max_tokens})!")
        
    return chunks