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
# pixel 22, 37, 52, 81, 91 are chosen by eye. 
22 : [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20,     23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 100, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112],
37 : [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36,     38, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 100, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112],
52 : [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51,     53, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 100, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112],
81 : [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,     86, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 100, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112],
91 : [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 86, 87, 88, 89, 90,     93, 94, 95, 96, 97, 98, 100, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112],
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
