import os

def get_files(datapath):
    """
    Reads all .edf files from the dataset structure:
    train/normal, train/abnormal, eval/normal, eval/abnormal
    """

    trainfiles = {"normal": [], "abnormal": []}
    evalfiles = {"normal": [], "abnormal": []}

    for root, dirs, files in os.walk(datapath):
        parts = os.path.normpath(root).split(os.sep)

        # We expect: [..., train/eval, normal/abnormal]
        if len(parts) < 3:
            continue

        parent = parts[-2].lower()   # train or eval
        child = parts[-1].lower()    # normal or abnormal

        if parent not in ["train", "eval"]:
            continue
        if child not in ["normal", "abnormal"]:
            continue

        for f in files:
            if f.endswith(".edf"):
                fpath = os.path.join(root, f)
                if parent == "train":
                    trainfiles[child].append(fpath)
                else:
                    evalfiles[child].append(fpath)

    return trainfiles, evalfiles


# Example usage
if __name__ == "__main__":
    datapath = "E:/NMT_events/NMT_events/edf"
    trainfiles, evalfiles = get_files(datapath)

    print("Training total:", sum(len(v) for v in trainfiles.values()))
    print("  normal:", len(trainfiles["normal"]))
    print("  abnormal:", len(trainfiles["abnormal"]))

    print("Evaluation total:", sum(len(v) for v in evalfiles.values()))
    print("  normal:", len(evalfiles["normal"]))
    print("  abnormal:", len(evalfiles["abnormal"]))
