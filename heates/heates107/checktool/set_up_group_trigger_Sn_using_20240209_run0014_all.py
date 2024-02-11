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
#
# gererated using  python heates107_check_chan.py 20240209_run0014.root 
# pixek 24, 32, 53, and 94 are chosen by eye. 
24 : [3, 5, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20, 21, 22, 23,     25, 26, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 64, 65, 66, 67, 71, 72, 73, 75, 76, 77, 79, 80, 82, 83, 86, 88, 89, 92, 93, 94, 96, 101, 102, 103, 105, 107, 109, 111],
32 : [3, 5, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 31,     33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 64, 65, 66, 67, 71, 72, 73, 75, 76, 77, 79, 80, 82, 83, 86, 88, 89, 92, 93, 94, 96, 101, 102, 103, 105, 107, 109, 111],
53 : [3, 5, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,     54, 55, 56, 57, 58, 60, 61, 62, 64, 65, 66, 67, 71, 72, 73, 75, 76, 77, 79, 80, 82, 83, 86, 88, 89, 92, 93, 94, 96, 101, 102, 103, 105, 107, 109, 111],
94 : [3, 5, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 64, 65, 66, 67, 71, 72, 73, 75, 76, 77, 79, 80, 82, 83, 86, 88, 89, 92, 93,     96, 101, 102, 103, 105, 107, 109, 111]
}

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
