import os
import subprocess
import glob
import json
import time
import sys
import urllib.request
import re
from multiprocessing.dummy import Pool

import random

import logging

logging.basicConfig(
    filename="videoDownloader.log", filemode="w", level=logging.DEBUG
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


class videoDownloader:
    wordCounts = {}

    def __init__(self, idxf="WLASL_v0.3.json", vd="data", n=1, m=2000000):
        self.wordCounts = {}
        self.indexFile = idxf
        self.max = m
        self.videoDir = vd
        if not os.path.exists(self.videoDir):
            os.mkdir(self.videoDir)
        self.size = self.updateSize()

    def updateSize(self):
        self.size = int(
            subprocess.check_output(["du", "-ks", self.videoDir])
            .split()[0]
            .decode("utf-8")
        )
        return self.size

    def request_video(self, url, referer=""):
        user_agent = (
            "Mozilla/5.0"
            "(Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7)"
            "Gecko/2009021910 Firefox/3.0.7"
        )

        headers = {
            "User-Agent": user_agent,
        }

        if referer:
            headers["Referer"] = referer

        # The assembled request
        request = urllib.request.Request(url, None, headers)

        logging.info("Requesting {}".format(url))
        response = urllib.request.urlopen(request)
        data = response.read()  # The data you need
        urllib.request.urlopen

        return data

    def dlPass(self, video_id):
        logging.info(f"Download Successful\t-\t{video_id}")

    def dlFail(self, video_id):
        logging.error(f"Download Failed\t\t-\t{video_id}")

    def download(self, inst, gloss):
        rv = False
        saveto = os.path.join(self.videoDir, gloss, inst["video_id"])
        if glob.glob(f"{saveto}.*"):
            logging.info(f"{inst['video_id']} exists at {saveto} - Skipping")
            rv = True
        else:
            if re.search(r"youtu\.?be", inst["url"]):
                status = os.system(
                    f"youtube-dl \"{inst['url']}\" -o \"{saveto}.yt.%(ext)s\""
                )
                if status == 0:
                    self.dlPass(inst["video_id"])
                    rv = True
                else:
                    rv = False
                    self.dlFail(inst["video_id"])
            else:
                if "aslpro" in inst["url"]:
                    saveto = f"{saveto}.swf"
                    ref = "http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi"
                else:
                    saveto = f"{saveto}.mp4"
                    ref = ""
                dat = self.request_video(inst["url"], referer=ref)
                if dat:
                    with open(saveto, "wb+") as f:
                        f.write(dat)
                        self.dlPass(inst["video_id"])
                        rv = True
                else:
                    self.dlFail(inst["video_id"])
            # please be nice to the host - take pauses and avoid spamming
            time.sleep(random.uniform(0.3, 0.7))
        return rv

    def main(self):
        idx = json.load(open(self.indexFile))
        idx = sorted(idx, key=lambda x: (len(x["instances"])), reverse=True)

        if not os.path.exists(self.videoDir):
            os.mkdir(self.videoDir)

        for i in idx:
            if not os.path.exists(os.path.join(self.videoDir, i["gloss"])):
                os.mkdir(os.path.join(self.videoDir, i["gloss"]))
            if i["gloss"] not in self.wordCounts:
                self.wordCounts[i["gloss"]] = 0
            if self.updateSize() >= self.max:
                logging.info("Max size reached")
                break
            for j in i["instances"]:
                if self.updateSize() >= self.max:
                    break
                logging.info(
                    f">>>GLOSS: {i['gloss']}"
                    f"\tvideo: {j['video_id']}"
                    f"\tcount: {self.wordCounts[i['gloss']]}"
                )
                try:
                    if self.download(j, i["gloss"]):
                        self.wordCounts[i["gloss"]] = (
                            self.wordCounts[i["gloss"]] + 1
                        )
                except Exception as e:
                    logging.error(f"ERROR - {j['video_id']}: {e}")


if __name__ == "__main__":
    vd = videoDownloader()
    vd.main()
