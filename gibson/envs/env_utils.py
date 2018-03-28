def get_segmentId_by_name(meta, name):
    category_ids = []
    object_ids = []
    segment_ids = []
    with open(meta) as f:
        for line in f:
            if line[:2] == "C " and name in line:
                c_index = int(line.split()[1])
                category_ids.append(c_index)
            if line[:2] == "O ":
                c_index = int(line.split()[3])
                o_index = int(line.split()[1])
                if c_index in category_ids:
                    object_ids.append(o_index)
            if line[:2] == "E ":
                o_index = int(line.split()[2])
                e_index = int(line.split()[1])
                if o_index in object_ids:
                    segment_ids.append(e_index)
    return category_ids, object_ids, segment_ids

if __name__ == "__main__":
    meta = "/home/zhiyang/Desktop/universe-test/GibsonEnv/gibson/assets/dataset/17DRP5sb8fy/semantic.house"
    c_id, o_id, e_id = get_segmentId_by_name(meta, "chair")
    print(len(c_id))
    print(len(o_id))
    print(len(e_id))
    print(o_id)