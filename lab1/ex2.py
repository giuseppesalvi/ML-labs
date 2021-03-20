import sys


if __name__ == '__main__':
    if len(sys.argv) == 4:
        file_name = sys.argv[1]
        flag = sys.argv[2]
        target = sys.argv[3]
        with open(file_name, 'r') as f:
            if flag == '-b':
                last_x = None
                last_y = None
                total = 0
                for line in f:
                    bus_id, line_id, x, y, time = line.split()
                    if bus_id == target:
                        if last_x is not None:
                            total += int(x) - int(last_x)
                            total += int(y) - int(last_y)
                        last_x = x
                        last_y = y
                print('%s - Total Distance: %d' % (target, total))
            elif flag == '-l':
                buses = {}  # a dictionary containing the buses in the line
                for line in f:
                    bus_id, line_id, x, y, time = line.split()
                    if line_id != target:
                        continue
                    if bus_id not in buses:
                        buses[bus_id] = {'tot_time': 0,
                                         'last_time': int(time),
                                         'tot_distance': 0,
                                         'last_x': int(x),
                                         'last_y': int(y)}
                    else:
                        time_add = (int(time) - buses[bus_id]['last_time'])
                        distance_add = (int(x) - buses[bus_id]['last_x'])
                        distance_add += (int(y) - buses[bus_id]['last_y'])
                        buses[bus_id]['tot_time'] += time_add
                        buses[bus_id]['tot_distance'] += distance_add
                # compute the avg speed of each bus and then the line avg
                speeds = 0.0
                number_buses = 0
                values = buses.values()
                for b in values:
                    tot_distance = b['tot_distance']
                    tot_time = b['tot_time']
                    speeds += tot_distance / tot_time
                    number_buses += 1
                # print("avg speed line" + speeds/number_buses)
                print("avg speed line %f" % (speeds/number_buses))
                # line_distance = 0.0
                # line_time = 0.0
                # values = buses.values()
                # for b in values:
                #     line_distance += b['tot_distance']
                #     line_time += b['tot_time']
                # print("avg speed line %f" % (line_distance/line_time))
            else:
                print("wrong parameter flag: must be -b or -l")
    else:
        print("wrong format")
