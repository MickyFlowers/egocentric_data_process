import os

def oss_to_local(oss_path: str, mnt_path: str="/home") -> str:
    assert oss_path.startswith("oss://"), "oss_path must start with 'oss://'"\
    
    tmp_path = oss_path.replace("oss://", '')
    oss_path = os.path.join(mnt_path, tmp_path)
    return oss_path

def local_to_oss(local_path: str, mnt_path: str="/home") -> str:
    assert local_path.startswith(mnt_path), "local_path must start with 'mnt_path'"
    if not mnt_path.endswith('/'):
        mnt_path += '/'
    oss_path = os.path.join("oss://", local_path.replace(mnt_path, ''))
    return oss_path


if __name__ == "__main__":
    # test oss_to_local
    print("input oss path: oss://bucket/path/to/file.txt")
    print("output local path: ", oss_to_local("oss://bucket/path/to/file.txt"))
    # test local_to_oss
    print("input local path: /home/user/file.txt")
    print("output oss path: ", local_to_oss("/home/user/file.txt"))

