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
# gererated using  python heates107_check_chan.py 20240209_run0014.root 
 3: [111, 5],
 5: [3, 7],
 7: [5, 8],
 8: [7, 9],
 9: [8, 10],
 10: [9, 11],
 11: [10, 12],
 12: [11, 15],
 15: [12, 16],
 16: [15, 18],
 18: [16, 19],
 19: [18, 20],
 20: [19, 21],
 21: [20, 22],
 22: [21, 23],
 23: [22, 24],
 24: [23, 25],
 25: [24, 26],
 26: [25, 29],
 29: [26, 31],
 31: [29, 32],
 32: [31, 33],
 33: [32, 34],
 34: [33, 35],
 35: [34, 36],
 36: [35, 37],
 37: [36, 38],
 38: [37, 39],
 39: [38, 40],
 40: [39, 41],
 41: [40, 43],
 43: [41, 44],
 44: [43, 45],
 45: [44, 46],
 46: [45, 47],
 47: [46, 48],
 48: [47, 49],
 49: [48, 50],
 50: [49, 51],
 51: [50, 52],
 52: [51, 53],
 53: [52, 54],
 54: [53, 55],
 55: [54, 56],
 56: [55, 57],
 57: [56, 58],
 58: [57, 60],
 60: [58, 61],
 61: [60, 62],
 62: [61, 64],
 64: [62, 65],
 65: [64, 66],
 66: [65, 67],
 67: [66, 71],
 71: [67, 72],
 72: [71, 73],
 73: [72, 75],
 75: [73, 76],
 76: [75, 77],
 77: [76, 79],
 79: [77, 80],
 80: [79, 82],
 82: [80, 83],
 83: [82, 86],
 86: [83, 88],
 88: [86, 89],
 89: [88, 92],
 92: [89, 93],
 93: [92, 94],
 94: [93, 96],
 96: [94, 101],
 101: [96, 102],
 102: [101, 103],
 103: [102, 105],
 105: [103, 107],
 107: [105, 109],
 109: [107, 111],
 111: [109, 3]}

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
