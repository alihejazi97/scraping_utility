from IPython.display import Audio, display
import os, glob, zipfile, subprocess
import chardet
from typing import Union
from pathlib import Path
import requests, urllib.request
import re, logging, json
from tqdm.notebook import tqdm
import editdistance

import srt
from srt import Subtitle
import numpy as np

import ffmpeg


opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWeb'),
            ('User-agent', 'Kit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safa'),
            ('User-agent', 'ri/537.36')]
urllib.request.install_opener(opener)

class ScrapeUtilityVariableStorage:
    SAMPLING_RATE : float = 16000
    PARANTHESIS_REGEX_STR_LIST : list[str] = [r'(\[[\S\s]*\])',r'(\][\S\s]*\[)',r'(\)[\S\s]*\()',r'(\([\S\s]*\))']
    PARANTHESIS_REGEX : re.Pattern = re.compile(r'|'.join(PARANTHESIS_REGEX_STR_LIST))
    LEADING_TRAILING_WHITESPACE_REGEX : re.Pattern = re.compile(r'^\s+|\s+$')
    CONTINOUS_SYMBOL_REGEX = re.compile(r'(^(\.\.\.))|(((\.\.\.)|:)$)')
    SUBDL_SEARCH_MOVIE_URL = 'https://api.subdl.com/auto'
    SUBDL_SEARCH_SUBTITLE_URL = 'https://api.subdl.com/api/v1/subtitles'
    SUBDL_SUBTITLE_DOWNLOAD_URL = 'https://dl.subdl.com'
    MIN_LENGTH_TO_RETURN : float = 0.2
    SUBDL_API_KEY : str = None
    SUBDL_LANG_CODES : dict [str, list[str]] = {
        'fa' : ['FA'],          # persian
        'en' : ['EN'],          # english
        'de' : ['DE', 'EN_DE'], # german
        'fr' : ['FR'],          # france
        'es' : ['ES'],          # spanish
        'it' : ['IT'],          # italian
        'ar' : ['AR'],          # arabic
        'pt' : ['PT'],          # portuguese
        'ru' : ['RU'],          # russian
        'tr' : ['TR'],          # turkish
        'uk' : ['UK'],          # ukrainian
        'pl' : ['PL'],          # polish
        'zh' : ['ZH_BG', 'ZH'], # chinese
        'hi' : ['HI'],          # hindi
        'bn' : ['BN'],          # bengali
        'ja' : ['JA'],          # japanese
        'vi' : ['VI'],          # vietnamese
        'ko' : ['KO'],          # korean
        'fi' : ['FI'],          # finnish
        'ur' : ['UR']           # Urdu
    }

    SUBDL_LANGUAGE_MAP : dict[str, str] = {
    "farsi_persian" : 'fa',     # persian
    "english" : 'en',           # english
    "english_german" : 'de',    # german
    "german" : 'de',            # german
    "french" : 'fr',            # france
    "spanish" : 'es',           # spanish
    "italian" : 'it',           # italian
    "arabic" : 'ar',            # arabic
    "portuguese" : 'pt',        # portuguese
    "russian" : 'ru',           # russian
    "turkish" : 'tr',           # turkish
    "ukranian" : 'uk',          # ukrainian
    "polish": 'pl',             # polish
    "big 5 code": 'zh',         # chinese
    "chinese bg code" : 'zh',   # chinese
    "hindi" : "hi",             # hindi
    "bengali" : "bn",           # bengali
    "japanese" : "ja",          # japanese
    "vietnamese" : "vi",        # vietnamese
    "korean" : "ko",            # korean
    "finnish" : "fi",           # finnish
    "urdu" : "ur"               # Urdu
    }

def save_json(path: Union[str, os.PathLike], obj):
    with open(path,  'w',  encoding='utf-8') as f:
        json.dump(obj,  f,  ensure_ascii=False)

def load_json(path: Union[str, os.PathLike]):
    with open(path, 'r') as f:
        _obj = json.load(f)
    return _obj

class ShowProgressUrllib:
    def __init__(self, description: str):
        self._pre_block_num = -1
        self._description = description
        self._tqdm_pbar = None

    def __call__(self, block_num, block_size, total_size):
        if self._tqdm_pbar is None:
            self._tqdm_pbar = tqdm(desc=self._description,total=(total_size), unit='B', unit_scale=True)

        _all_downloaded_bytes = (block_num * block_size)
        _current_downloaded_bytes = ((block_num - self._pre_block_num) * block_size)
        self._pre_block_num = block_num
        if _all_downloaded_bytes < total_size:
            self._tqdm_pbar.update(_current_downloaded_bytes)
        else:
            self._tqdm_pbar.clear()
            self._tqdm_pbar.close()
            self._pre_block_num = -1

def create_directory(directory: Union[str, os.PathLike]):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_segment_from_array(audio_array: np.array, sample_rate: int, subtitle: Subtitle, min_length_to_return: float = 0.2):
    _segment_start_index = int(subtitle.start.total_seconds()*sample_rate)
    _segment_end_index = int(subtitle.end.total_seconds()*sample_rate)
    _segment_array = audio_array[_segment_start_index: _segment_end_index]
    _min_length = int(min_length_to_return * sample_rate)
    if _segment_array.shape[0] > _min_length:
        return _segment_array
    else:
        return np.zeros(shape=(_min_length,), dtype=np.float32)

def show_parallel_data(audio_array: np.array, sample_rate: int, subtitle):
    print(f'subtitle : {subtitle.content}')
    _audio_array = extract_segment_from_array(audio_array, sample_rate, subtitle)
    display(Audio(_audio_array, rate=sample_rate))

def get_min_download_bar(movie, encoder : str = None):
    _minsize : int = 999_999_999
    _minsize_item = None
    _minsize_index : int = 0
    for _minsize_idx, _download_bar in enumerate(movie['download_results'][0]['download_bars']):
        if _download_bar['size'] < _minsize:
            if encoder:
                if 'encoder' not in _download_bar:
                    continue
                if not (_download_bar['encoder'].lower() == encoder.lower()):
                    continue
            _minsize = _download_bar['size']
            _minsize_item = _download_bar
            _minsize_index = _minsize_idx
    return _minsize_item, _minsize_index

def get_best_download_bar(movie):
    _minsize_item, _minsize_index = get_min_download_bar(movie)
    _minsize_item_encoder_30nama, _minsize_index_encoder_30nama = get_min_download_bar(movie, '30nama')
    if _minsize_item_encoder_30nama:
        return _minsize_index_encoder_30nama, _minsize_item_encoder_30nama
    return _minsize_index, _minsize_item

def find_movies_subdl(title: str, year: int):
    _payload = {'query' : title}
    _response = requests.get(ScrapeUtilityVariableStorage.SUBDL_SEARCH_MOVIE_URL, params=_payload).json()
    for _posible_movie in _response['results']:
        _posible_movie['score'] = editdistance.eval(title, _posible_movie['name'])
        if _posible_movie['year'] == year:
            _posible_movie['score'] = _posible_movie['score'] - 5
    _response['results'].sort(key=lambda x: x['score'])
    return _response['results']

class SubdlException(Exception):
    def __init__(self, message) -> None:
        super(Exception, self).__init__(message)
 
class ScarapingException(Exception):
    def __init__(self, message) -> None:
        super(Exception, self).__init__(message)

class CNamaDownloadException(Exception):
    def __init__(self, message) -> None:
        super(Exception, self).__init__(message)

def find_subtitle(title: str, year: int, languages: list[str]):
    _languages_subdl = ','.join(languages)
    _payload = {'api_key': ScrapeUtilityVariableStorage.SUBDL_API_KEY, 'film_name' : title, 'languages' : _languages_subdl}
    if type(year) == int and year >= 1800 and year <= 2024:
        _payload['year'] = year
    _response = requests.get(ScrapeUtilityVariableStorage.SUBDL_SEARCH_SUBTITLE_URL, params=_payload).json()
    if _response['status'] == False:
        raise SubdlException(f'Subdl result status was false')
    else:
        return _response['subtitles']

def extract_title_year_from_30nama_title(movie):
    if len(movie['title']) < 6:
        raise ScarapingException('movie title is too short. we can not extract title and year from it.')
    _title_subdl = movie['title'][:-5]
    try:
        _year_subdl = int(movie['title'][-4:])
    except ValueError:
        raise ScarapingException('year can not be converted to integer.')
    return _title_subdl, _year_subdl

def download_subtitle_subdl(movie, languages: list[str], processed_langs: list[str], max_subtitle_movie_try: int):
    _movie_id = movie['id']
    _title_subdl, _year_subdl = extract_title_year_from_30nama_title(movie)
    _posible_movies = find_movies_subdl(_title_subdl, _year_subdl)
    for idx_posible_movie_id, _posible_movie in enumerate(_posible_movies[:max_subtitle_movie_try]):
        _subdl_subtitles = find_subtitle(_posible_movie['name'] ,_posible_movie['year'], languages)
        for _subtitle_id, _subtitle in enumerate(_subdl_subtitles):
            if _subtitle['lang'].lower() in ScrapeUtilityVariableStorage.SUBDL_LANGUAGE_MAP:
                _subtitle_lang = ScrapeUtilityVariableStorage.SUBDL_LANGUAGE_MAP[_subtitle['lang'].lower()]
            else:
                logging.warn(f"this language({_subtitle['lang'].lower()}) is  not supported in our snippet but was returned by the api.")
                continue
            if _subtitle_lang in processed_langs:
                continue
            if (_subtitle_lang not in languages):
                logging.warn(f"this language ({_subtitle['lang'].lower()}) is not wanted but is in the subdl api results.\npossible movie: {_posible_movie['name']}\nspossible movie year{_posible_movie['year']}\n returned subtitle object : {_subtitle}\n")
                continue
            create_directory(f'./temp/{_movie_id}/')
            if not _subtitle['url']:
                raise Exception(f"empty subtitle url {_subtitle['url']} is not wanted but is in the subdl api results.")
            _zip_path = f'./temp/{_movie_id}/sub_{idx_posible_movie_id}_{_subtitle_id}_{_subtitle_lang}.zip'
            if os.path.isfile(_zip_path):
                continue
            _zipped_url = ScrapeUtilityVariableStorage.SUBDL_SUBTITLE_DOWNLOAD_URL + '/' + _subtitle['url']
            try:
                urllib.request.urlretrieve(_zipped_url,_zip_path)
            except Exception:
                raise SubdlException(f'can not retrive following subd subtitle :{_zipped_url}')
            with zipfile.ZipFile(_zip_path, 'r') as _zip_ref:
                _zip_ref.extractall(f'./temp/{_movie_id}/sub_{idx_posible_movie_id}_{_subtitle_id}_{_subtitle_lang}/')
                _srt_list = glob.glob(f'./temp/{_movie_id}/sub_{idx_posible_movie_id}_{_subtitle_id}_{_subtitle_lang}/?*.srt')
                for _subtitle_path in _srt_list:
                    yield _subtitle_path, _subtitle_lang

def download_video_file(download_bar, download_bar_index, movie, sampling_rate : float, audio_path):
    movie_id = movie['id']
    if not os.path.isfile(audio_path + 'all.wav'):
        _download_link = download_bar['download_link']
        _movie_extention = _download_link.split('/')[-1].split('.')[-1]
        _video_file_path = f'./temp/{movie_id}/{download_bar_index}.{_movie_extention}'
        try:
            urllib.request.urlretrieve(_download_link, _video_file_path, ShowProgressUrllib())
        except Exception:
            CNamaDownloadException('Today maximum requests have reached its limits.')
        ffmpeg.input(_video_file_path).output(f'./dataset/{movie_id}/all.mp3', ac=1, ar=sampling_rate).run()
        return True

def load_subtitle_file(subtitle_path):
    _subtitle_encoding  = predict_encoding(subtitle_path)
    try:
        _file = open(subtitle_path, encoding=_subtitle_encoding)
        _file_content = _file.read()
        _srt_parsed_generator = srt.parse(_file_content)
        return list(_srt_parsed_generator)
    except Exception as e:
        logging.warn(f'subtitle can not be procdessed.\nsubtitle_path: {subtitle_path}\nsubtitle predicted encoding: {_subtitle_encoding}')
        return []

def sybc_and_load_subtitle_file(subtitle_path, movie):
    _movie_id = movie['id']
    _audio_path = f'./dataset/{_movie_id}/all.mp3'
    _sync_path = '/'.join(subtitle_path.split('/')[:-1]) + '/sync.srt'
    subprocess.run(["ffs", _audio_path, '-i', subtitle_path, '-o', _sync_path])
    if os.path.exists(_sync_path):
        return load_subtitle_file(_sync_path), _sync_path
    else:
        logging.warn(f'ffs can not sync the subtitle.\nsubtitle_path: {subtitle_path}')
        return [] , ''

def preprocess_subtitle_str(subtitle_str):
    _subtitle_str = subtitle_str.replace('\u200c',' ')
    _subtitle_str = _subtitle_str.replace('\n',' ')
    _subtitle_str = ScrapeUtilityVariableStorage.LEADING_TRAILING_WHITESPACE_REGEX.sub('', _subtitle_str)
    _subtitle_str = ScrapeUtilityVariableStorage.PARANTHESIS_REGEX.sub('', _subtitle_str)
    _subtitle_str = ScrapeUtilityVariableStorage.CONTINOUS_SYMBOL_REGEX.sub('', _subtitle_str)
    _subtitle_str = ScrapeUtilityVariableStorage.LEADING_TRAILING_WHITESPACE_REGEX.sub('', _subtitle_str)
    return _subtitle_str

def zip_wavs(movie):
    _movie_id = movie['id']
    with zipfile.ZipFile(f'./dataset/{_movie_id}/{_movie_id}.zip', 'w', zipfile.ZIP_DEFLATED) as _zipf:
        _wavs = glob.glob(f'./dataset/{_movie_id}/*.mp3')
        for _wav in _wavs:
            _zipf.write(_wav, _wav.split('/')[-1], compress_type=zipfile.ZIP_DEFLATED)
            os.remove(_wav)

def get_movie_str(movie):
    return f'movie title: {movie["title"]} ### movie id: {movie["idgr"]}'

def check_movie(movie):
    if movie['is_series']:
        return False
    if 'subtitle_results' not in movie:
        return False
    if 'download_results' not in movie:
        return False
    if len(movie['download_results']) == 0:
        logging.info(f"{get_movie_str(movie)}\nbecause movie has 0 download_results it was dropped.")
        return False
    if len(movie['download_results']) < 1:
        logging.info(f"{get_movie_str(movie)}\nbeacause movie has more than 1 download_results it was dropped.")
        return False
    if len(movie['download_results'][0]) == 0:
        logging.info(f"{get_movie_str(movie)}\nbeacause movie has 0 download bars it was dropped.")
        return False
    # if len(movie['subtitle_results']) == 0:
    #     logging.info(f"{get_movie_str(movie)}\nbeacause movie has 0 subtitle it was dropped.")
    #     return False
    return True

def predict_encoding(file_path: Path, n_lines: int=20) -> str:
    '''Predict a file's encoding using chardet'''

    if file_path.find('[ANSI]') != -1:
        return 'cp1256'

    # Open the file as binary data
    with Path(file_path).open('rb') as f:
        # Join binary lines for specified number of lines
        _rawdata = b''.join([f.readline() for _ in range(n_lines)])

    return chardet.detect(_rawdata)['encoding']