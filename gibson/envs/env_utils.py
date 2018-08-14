
def get_segmentId_by_name_MP3D(meta, name):
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


def get_segmentId_by_name_2D3DS(mtl, obj, name):
    semantic_label_list = get_semantic_label_list_2D3DS(obj)
    object_ids = []
    obj_id = 0
    '''with open(mtl) as f:
        for line in f:
            if "newmtl" in line:
                semantic_label = line.split()[1]
                object_name, object_counchairt, segment_count = semantic_label.split("_")
                if name in object_name:
                    object_ids.append(obj_id)
                obj_id += 1
    '''
    for semantic_label in semantic_label_list:
        assert len(semantic_label.split("_")) == 5 or len(semantic_label.split("_")) == 3, "Unable to parse semantic label {}".format(semantic_label)
        if len(semantic_label.split("_")) == 5:
            object_name, object_count, context_name, context_count, floor_count = semantic_label.split("_")
            if name in object_name:
                object_ids.append(obj_id)
            obj_id += 1 
        if len(semantic_label.split("_")) == 3:
            object_name, object_count, floor_count = semantic_label.split("_")
            if name in object_name:
                object_ids.append(obj_id)
            obj_id += 1
    return None, object_ids, None


def get_semantic_label_list_2D3DS(obj):
    label_list = []
    with open(obj) as f:
        for line in f:
            if "usemtl " in line:
                label_list.append(line.split()[1])
    return label_list

if __name__ == "__main__":
    meta = ""
    #meta = "/home/zhiyang/Desktop/universe-test/GibsonEnv/gibson/assets/dataset/17DRP5sb8fy/semantic.house"
    c_id, o_id, e_id = get_segmentId_by_name(meta, "chair")
    print(len(c_id))
    print(len(o_id))
    print(len(e_id))
    print(o_id)