from __future__ import print_function
import os, traceback, sys
import argparse
import warnings
import json
from scapy.all import Dot11ProbeReq, Dot11,sniff, Dot11ProbeResp, TCP
import manuf
from backend.sensor import upload_wifi_info

def monitorDevice(pkt):
    addr, target = None, None
    time_stamp = -1
    if pkt.haslayer(Dot11ProbeReq):
        addr = pkt.getlayer(Dot11).addr2
        target = pkt.getlayer(Dot11ProbeReq).info or ''
        time_stamp = pkt.getlayer(Dot11ProbeReq).time
    if pkt.haslayer(Dot11ProbeResp):
        addr = pkt.getlayer(Dot11).addr1
        target = pkt.getlayer(Dot11ProbeResp).info or ''
        time_stamp = pkt.getlayer(Dot11ProbeResp).time
    if addr and target:
        manuf = parser.get_manuf(addr) or 'Unknown'
        rssi = -(256-ord(pkt.notdecoded[-4:-3]))
        try:
            print('Detected Devices: MAC[%s] Manuf[%s] Target[%s] RSSI [%d]' % (addr, manuf, target, rssi))
            upload_wifi_info(addr, rssi, int(conf['sensorId']), time_stamp)
            # print('Detected Devices: MAC[%s] Target[%s] RSSI [%d]' % (addr, target, rssi))
        except Exception, err:
            traceback.print_exc()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--conf", required=True, help="json config file path")
    args = vars(ap.parse_args())
    warnings.filterwarnings("ignore")
    conf = json.load(open(args["conf"]))

    interface = 'wlan1'
    parser = manuf.MacParser()
    sniff(iface=interface, prn=monitorDevice, store=0)
