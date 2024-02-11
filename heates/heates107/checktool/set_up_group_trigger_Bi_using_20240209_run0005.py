#!/usr/bin/env python
"""
set_up_group_trigger.py

An example program that can connect to a Dastard server as a short-lived client
to set up group triggering.

Suggested use: make a copy of this file and edit the variables just below this doc
string to suit your needs.
"""

from dastardcommander import rpc_client

# ----- <configuration> -----
# Configure the script by changing these variables.
Dastard_host = "localhost"
Dastard_port = "5500"
Connections = {
# In this example, triggers on channel 1 cause secondary triggers on 3, 4, or 5,
# and triggers on channel 2 cause secondaries on 5, 10, 15, or 20.
#    1: [3, 4, 5],
#    2: [5, 10, 15, 20]
#}
# gererated using  python heates107_check_chan.py 20240209_run0005.root 
 1: [112, 2],
 2: [1, 3],
 3: [2, 4],
 4: [3, 5],
 5: [4, 7],
 7: [5, 8],
 8: [7, 9],
 9: [8, 10],
 10: [9, 11],
 11: [10, 12],
 12: [11, 13],
 13: [12, 15],
 15: [13, 17],
 17: [15, 18],
 18: [17, 19],
 19: [18, 20],
 20: [19, 22],
 22: [20, 23],
 23: [22, 24],
 24: [23, 25],
 25: [24, 26],
 26: [25, 27],
 27: [26, 30],
 30: [27, 31],
 31: [30, 32],
 32: [31, 33],
 33: [32, 34],
 34: [33, 35],
 35: [34, 36],
 36: [35, 37],
 37: [36, 38],
 38: [37, 41],
 41: [38, 42],
 42: [41, 43],
 43: [42, 44],
 44: [43, 45],
 45: [44, 46],
 46: [45, 48],
 48: [46, 49],
 49: [48, 50],
 50: [49, 51],
 51: [50, 52],
 52: [51, 53],
 53: [52, 57],
 57: [53, 58],
 58: [57, 59],
 59: [58, 60],
 60: [59, 61],
 61: [60, 62],
 62: [61, 63],
 63: [62, 64],
 64: [63, 65],
 65: [64, 66],
 66: [65, 67],
 67: [66, 68],
 68: [67, 69],
 69: [68, 71],
 71: [69, 72],
 72: [71, 73],
 73: [72, 74],
 74: [73, 75],
 75: [74, 76],
 76: [75, 77],
 77: [76, 78],
 78: [77, 79],
 79: [78, 80],
 80: [79, 81],
 81: [80, 86],
 86: [81, 87],
 87: [86, 88],
 88: [87, 89],
 89: [88, 90],
 90: [89, 91],
 91: [90, 93],
 93: [91, 94],
 94: [93, 95],
 95: [94, 96],
 96: [95, 97],
 97: [96, 98],
 98: [97, 100],
 100: [98, 102],
 102: [100, 103],
 103: [102, 104],
 104: [103, 105],
 105: [104, 107],
 107: [105, 108],
 108: [107, 109],
 109: [108, 110],
 110: [109, 111],
 111: [110, 112],
 112: [111, 1]}

Add_connections = True
Complete_disconnect = False
# ----- </configuration> -----


def main():
    client = rpc_client.JSONClient((Dastard_host, Dastard_port))
    try:
        if Complete_disconnect:
            dummy = True
            client.call("SourceControl.StopTriggerCoupling", dummy)
            print("Successfully disconnected all group trigger couplings")
            return

        state = {"Connections": Connections}
        request = "SourceControl.AddGroupTriggerCoupling"
        action = "added"
        if not Add_connections:
            request = "SourceControl.DeleteGroupTriggerCoupling"
            action = "deleted"
        client.call(request, state)
        print("Successfully {} group trigger couplings {}".format(action, Connections))

    finally:
        client.close()


if __name__ == "__main__":
    main()
