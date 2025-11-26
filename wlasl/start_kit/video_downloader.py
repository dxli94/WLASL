import os, json, time, sys, random, logging, glob
import urllib.request
import pandas as pd
from multiprocessing.dummy import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set this to youtube-dl if you want to use youtube-dl.
# The the README for an explanation regarding yt-dlp vs youtube-dl.
youtube_downloader = "yt-dlp"

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


def download_nonyt_videos(saveto, gloss, video_id, video_url):
    logging.info('gloss: {}, video: {}.'.format(gloss, video_id))

    download_method = select_download_method(video_url)
    if download_method == download_youtube:
        logging.warning('Skipping YouTube video {}'.format(video_id))
    else:
        try:
            download_method(video_url, saveto, video_id)
        except Exception as e:
            logging.error('Unsuccessful downloading - video {}'.format(video_id))


def check_youtube_dl_version():
    ver = os.popen(f'{youtube_downloader} --version').read()

    assert ver, f"{youtube_downloader} cannot be found in PATH. Please verify your installation."


def download_yt_videos(saveto, video_url):
    if os.path.exists(os.path.join(saveto, video_url[-11:] + '.mp4')) or os.path.exists(os.path.join(saveto, video_url[-11:] + '.mkv')):
        logging.info('YouTube videos {} already exists.'.format(video_url))
    else:
        cmd = f"{youtube_downloader} \"{{}}\" -o \"{{}}%(id)s.%(ext)s\""
        cmd = cmd.format(video_url, saveto + os.path.sep)

        rv = os.system(cmd)
        
        if not rv:
            logging.info('Finish downloading youtube video url {}'.format(video_url))
        else:
            logging.error('Unsuccessful downloading - youtube video url {}'.format(video_url))

        # please be nice to the host - take pauses and avoid spamming
        time.sleep(random.uniform(1.0, 1.5))
    

def process_video(gloss, video_url, video_id, saveto):
    if 'youtube' not in video_url and 'youtu.be' not in video_url:
        download_nonyt_videos(saveto, gloss, video_id, video_url)
        pass
    else:
        download_yt_videos(saveto, video_url)


def get_latest_log(log_dir):
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    if not log_files:
        return None
    latest_log = max(log_files, key=os.path.getmtime)

    return latest_log


def get_skip_counter(indexfile, logfile):
    skip_counter = 0
    df = pd.read_json(indexfile)
    df_exploded = df.explode("instances", ignore_index=True)
    df_flat = pd.concat([df_exploded.drop(columns=["instances"]),pd.json_normalize(df_exploded["instances"])],axis=1)

    # read the log file and get the index of latest video_id
    with open(logfile, "r") as f:
        try:
            lines = f.readlines()

            log_df = pd.DataFrame({"raw": [line.strip() for line in lines]})
            log_df["video_id"] = log_df["raw"].str.extract(r"video:?\s*(\d+)")
            index = df_flat[df_flat["video_id"] == log_df['video_id'].dropna().iloc[-1]].index[0]

            if index > 0:
                skip_counter = index

        except Exception as e:
                logging.error(f"Error: {e}")

    return skip_counter


if __name__ == '__main__':
    indexfile = 'WLASL_v0.3.json'
    saveto = 'raw_videos'
    
    os.makedirs(saveto, exist_ok=True)
    content = json.load(open(indexfile))
    LOGFILE = get_latest_log(os.getcwd())

    logging.basicConfig(filename='download_{}.log'.format(int(time.time())), filemode='w', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('Start downloading videos using multithreading.')

    MAX_WORKERS = os.cpu_count() - 4
    futures = []
    count = 0
    skip_counter = get_skip_counter(indexfile, LOGFILE) - (MAX_WORKERS*2) if LOGFILE else 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for entry in content:
            gloss = entry['gloss']
            for inst in entry['instances']:
                if count >= skip_counter:
                    video_url = inst['url']
                    video_id = inst['video_id']
                    futures.append(executor.submit(process_video, gloss, video_url, video_id, saveto))

                count += 1

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error: {e}")

    logging.info('All downloads completed.')