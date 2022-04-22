import argparse
import logging
from collections import Counter

# 通过sentencepiece进行subword的处理
# 理念其实是类似n-gram的 用出现频率多个几个组成搭配的概念

def main(args):

    if args.mode == "subword":
        logging.warn("Subword model is based on `sentencepiece`.")

        import sentencepiece as splib

        cmd = ("--input={} --model_prefix={} --model_type=bpe "
               "--vocab_size={} --character_coverage={} "
               "--pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1 "
               "--eos_piece=<eos> --remove_extra_whitespaces=true".format(
                   args.input_file, args.output_file,
                   args.vocab_size, args.character_coverage))
        # 使用sentencepiece进行处理sub-word

        splib.SentencePieceTrainer.Train(cmd)
    else:
        with open(args.input_file, "r") as f:
            lines = [line.strip("\r\n ") for line in f]
        counter = Counter()
        # 使用计数方式处理sub-word
        if args.mode == "word":
            for line in lines:
                counter.update(line.split())
            # In word mode, vocab_list is sorted by frequency
            # Only selected top `vocab_size` vocabularies
            vocab_list = sorted(
                counter.keys(), key=lambda k: counter[k], reverse=True)[:args.vocab_size]
            # vocab_list使用word出现的频率进行排序 只保存排序最高vocab_size的word
        elif args.mode == "character":
            for line in lines:
                counter.update(line)
            # In character mode, vocab_list is sorted in alphabetical order
            vocab_list = sorted(counter)
            # character模式使用字母顺序排序 同样是使用计数方式进行

        logging.info("Collected totally {} vocabularies.".format(len(counter)))
        logging.info("Selected {} vocabularies.".format(len(vocab_list)))

        with open(args.output_file, "w") as f:
            f.write("\n".join(vocab_list))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        "Utility script to generate `vocab_file` needed by text encoder.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument(
        "--mode", choices=["character", "word", "subword"], required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--character_coverage", type=float, default=1)

    args = parser.parse_args()

    if args.mode != "subword":
        logging.warn(
            "`character_coverage` is not used in `word` and `character` mode.")
    if args.mode == "character":
        logging.warn("`vocab_size` is not used in `character` mode.")

    main(args)
