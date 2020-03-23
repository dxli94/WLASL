import os
import json
import time
import sys
import urllib.request
from multiprocessing.dummy import Pool

import random

import logging
logging.basicConfig(filename='download_{}.log'.format(int(time.time())), filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def request_video(url, referer=''):
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

    headers = {'User-Agent': user_agent,
               }
    
    if referer:
        headers['Referer'] = referer

    request = urllib.request.Request(url, None, headers)  # The assembled request

    logging.info('Requesting {}'.format(url))
    response = urllib.request.urlopen(request)
    data = response.read()  # The data you need

    return data


def save_video(data, saveto):
    with open(saveto, 'wb+') as f:
        f.write(data)

    # please be nice to the host - take pauses and avoid spamming
    time.sleep(random.uniform(0.5, 1.5))


def download_youtube(url, dirname, video_id):
    raise NotImplementedError("Urllib cannot deal with YouTube links.")


def download_aslpro(url, dirname, video_id):
    saveto = os.path.join(dirname, '{}.swf'.format(video_id))
    if os.path.exists(saveto):
        logging.info('{} exists at {}'.format(video_id, saveto))
        return 

    data = request_video(url, referer='http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi')
    save_video(data, saveto)


def download_others(url, dirname, video_id):
    saveto = os.path.join(dirname, '{}.mp4'.format(video_id))
    if os.path.exists(saveto):
        logging.info('{} exists at {}'.format(video_id, saveto))
        return 
    
    data = request_video(url)
    save_video(data, saveto)


def select_download_method(url):
    if 'aslpro' in url:
        return download_aslpro
    elif 'youtube' in url or 'youtu.be' in url:
        return download_youtube
    else:
        return download_others


def download_nonyt_videos(indexfile, saveto='raw_videos'):
    content = json.load(open(indexfile))

    if not os.path.exists(saveto):
        os.mkdir(saveto)

    for entry in content:
        gloss = entry['gloss']
        instances = entry['instances']

        for inst in instances:
            video_url = inst['url']
            video_id = inst['video_id']
            
            logging.info('gloss: {}, video: {}.'.format(gloss, video_id))

            download_method = select_download_method(video_url)    
            
            if download_method == download_youtube:
                logging.warning('Skipping YouTube video {}'.format(video_id))
                continue

            try:
                download_method(video_url, saveto, video_id)
            except Exception as e:
                logging.error('Unsuccessful downloading - video {}'.format(video_id))


def check_youtube_dl_version():
    ver = os.popen('youtube-dl --version').read()

    assert ver, "youtube-dl cannot be found in PATH. Please verify your installation."
    assert ver >= '2020.03.08', "Please update youtube-dl to newest version."


def download_yt_videos(indexfile, saveto='raw_videos'):
    content = json.load(open(indexfile))
    
    if not os.path.exists(saveto):
        os.mkdir(saveto)
    
    for entry in content:
        gloss = entry['gloss']
        instances = entry['instances']

        for inst in instances:
            video_url = inst['url']
            video_id = inst['video_id']

            if 'youtube' not in video_url and 'youtu.be' not in video_url:
                continue

            if os.path.exists(os.path.join(saveto, video_url[-11:] + '.mp4')) or os.path.exists(os.path.join(saveto, video_url[-11:] + '.mkv')):
                logging.info('YouTube videos {} already exists.'.format(video_url))
                continue
            else:
                cmd = "youtube-dl \"{}\" -o \"{}%(id)s.%(ext)s\""
                cmd = cmd.format(video_url, saveto + os.path.sep)

                rv = os.system(cmd)
                
                if not rv:
                    logging.info('Finish downloading youtube video url {}'.format(video_url))
                else:
                    logging.error('Unsuccessful downloading - youtube video url {}'.format(video_url))

                # please be nice to the host - take pauses and avoid spamming
                time.sleep(random.uniform(1.0, 1.5))
    

if __name__ == '__main__':
    logging.info('Start downloading non-youtube videos.')
    download_nonyt_videos('WLASL_v0.3.json')

    check_youtube_dl_version()
    logging.info('Start downloading youtube videos.')
    download_yt_videos('WLASL_v0.3.json')

