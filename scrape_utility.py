import os, zipfile, subprocess
from typing import Union
import requests, urllib.request
import logging
import editdistance
from general_utility import predict_encoding, ShowProgressUrllib, Movie
from settings import Settings
from iso639 import Language
from typing import List
from dataclasses import dataclass
import pysubs2

def find_subtitle(title: str, year: int, languages: List[Language], subdl_api_key: str):
    _languages_subdl = ','.join(list(map(Settings.SUBDL_LANG_CODES, languages)))
    _payload = {'api_key': subdl_api_key,
                'film_name' : title,
                'languages' : _languages_subdl}
    if type(year) == int and year >= 1800 and year <= 2024:
        _payload['year'] = year   
    try:
        _response = requests.get(Settings.SUBDL_SEARCH_SUBTITLE_URL, params=_payload)
        if _response.status_code != 200:
            raise SubdlException(f'Subdl "find subtitle api" returned status code was {_response.status_code}')
        _response_json = _response.json()
        if _response_json['status'] == False:
            logging.warning(f'Subdl "find subtitle api" has no result(status code is false) for movie={title} ### year={year}')
            return []
        else:
            return _response_json['subtitles']
    except requests.RequestsJSONDecodeError:
        raise SubdlException(f'can not convert Subdl "find subtitle api" results to json {_response.status_code}')

def sort_posible_movies(response_json, title: str, year: int):
    for _posible_movie in response_json['results']:
        _posible_movie['score'] = editdistance.eval(title, _posible_movie['name'])
        if _posible_movie['year'] == year:
            _posible_movie['score'] = _posible_movie['score'] - 5
    response_json['results'].sort(key=lambda x: x['score'])

def find_movies_subdl(title: str):
    _payload = {'query' : title}
    try:
        _response = requests.get(Settings.SUBDL_SEARCH_MOVIE_URL, params=_payload)
        if _response.status_code != 200:
            raise SubdlException(f'Subdl "find subtitle api" returned status code was {_response.status_code}')
        _response_json = _response.json()
        return _response_json
    except requests.RequestsJSONDecodeError:
        raise SubdlException(f'can not convert Subdl "find subtitle api" results to json {_response.status_code}')

def download_subtitle_subdl(movie: Movie, languages: list[Language], processed_langs: list[Language]=[],
                            subtitle_zip_directory: Union[str, os.PathLike] = './', subtitle_directory: Union[str, os.PathLike] = './',
                            max_posible_movie: int = 1):
    _title_subdl, _year_subdl = movie.extract_title_year_from_30nama_title()
    _posible_movies = find_movies_subdl(_title_subdl, _year_subdl)[:max_posible_movie]
    
    for idx_posible_movie_id, _posible_movie in enumerate(_posible_movies):
        _subdl_subtitles = find_subtitle(_posible_movie['name'] ,_posible_movie['year'], languages)
    
        for _subtitle_id, _subtitle in enumerate(_subdl_subtitles):
            if _subtitle['lang'].lower() in Settings.SUBDL_LANGUAGE_MAP:
                _subtitle_lang = Settings.SUBDL_LANGUAGE_MAP[_subtitle['lang'].lower()]
            else:
                _subtitle_lang = Settings.UNKNOWN_LANGUAGE
            if _subtitle_lang in processed_langs:
                continue
            if not _subtitle['url']:
                logging.warn(f"this subtitle has no url.\npossible movie: {_posible_movie['name']}\nspossible movie year{_posible_movie['year']}\n returned subtitle object : {_subtitle}\n")
                continue
            _zip_path = os.path.join(subtitle_zip_directory,f'sub_{idx_posible_movie_id}_{_subtitle_id}_{_subtitle_lang.part1}.zip')
            
            # don't download if file is already downloaded
            if not os.path.isfile(_zip_path):
                _zipped_url = Settings.SUBDL_SUBTITLE_DOWNLOAD_URL + '/' + _subtitle['url']
                try:
                    urllib.request.urlretrieve(_zipped_url,_zip_path)
                except Exception:
                    raise SubdlException(f'can not retrive following subd subtitle :{_zipped_url}')
            
            # extracting zip file
            with zipfile.ZipFile(_zip_path, 'r') as _zip_ref:
                _subtitle_members = list(map(lambda x: x[0],filter(lambda x: x[1] in pysubs2.formats.FILE_EXTENSION_TO_FORMAT_IDENTIFIER,
                    map(lambda x: (x,x.split('.')[-1],), _zip_ref.namelist()))))
                _zip_ref.extractall(subtitle_directory, members=_subtitle_members)
                for _subtitle_member in _subtitle_members:
                    yield os.path.join(subtitle_directory,_subtitle_member), _subtitle_lang


def download_video_file(download_link: str, video_title: str = '', video_file_directory: Union[str, os.PathLike] = './',
                        video_file_name: str = None) -> str:
    if not os.path.isdir(video_file_directory):
        raise VideoDownloadException(f'video file direcotry({video_file_directory}) does not exists.')
    _original_extention = download_link.split('/')[-1].split('.')[-1]
    _original_file_name = download_link.split('/')[-1]
    if video_file_name:
        _video_file_path = os.path.join(video_file_directory, video_file_name + '.' + _original_extention)
    else:
        _video_file_path = os.path.join(video_file_directory, _original_file_name)
    if os.path.exists(_video_file_path):
        return _video_file_path
    try:
        urllib.request.urlretrieve(download_link, _video_file_path, ShowProgressUrllib(f'downloading video {video_title}'))
        return _video_file_path
    except Exception:
        raise VideoDownloadException('Today maximum requests have reached its limits.')


def load_subtitle_file(subtitle_path: Union[str, os.PathLike]) -> pysubs2.SSAFile:
    _subtitle_encoding  = predict_encoding(subtitle_path)
    try:
        subtitlles = pysubs2.load(subtitle_path, _subtitle_encoding)
        return subtitlles
    except:
        logging.warn(f'subtitle can not be procdessed.\nsubtitle_path: {subtitle_path}\nsubtitle predicted encoding: {_subtitle_encoding}')
        return pysubs2.SSAFile()

def sync_subtitle_file(audio_path: Union[str, os.PathLike],subtitle_path: Union[str, os.PathLike], sync_path: Union[str, os.PathLike]):
    subprocess.run(["ffs", audio_path, '-i', subtitle_path, '-o', sync_path])
    if os.path.exists(sync_path):
        return sync_path
    else:
        logging.warn(f'ffs can not sync the subtitle.\nsubtitle_path: {subtitle_path}')
        return None

@dataclass
class SubtitleData:
    language: Language
    subtitle_url: str
    original_path: str
    sync_path: str
    movie_id: int

class SubdlException(Exception):
    def __init__(self, message) -> None:
        super(Exception, self).__init__(message)
 
class ScarapingException(Exception):
    def __init__(self, message) -> None:
        super(Exception, self).__init__(message)

class VideoDownloadException(Exception):
    def __init__(self, message) -> None:
        super(Exception, self).__init__(message)