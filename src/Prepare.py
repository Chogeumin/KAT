from konlpy.tag import Hannanum
from konlpy.tag import Kkma
from konlpy.tag import Twitter
from konlpy.tag import Komoran
from konlpy.tag import Mecab
import glob, sys

def kor_tokenize(tokenizer, indir, outdir):
    file_cnt = 1
    file_list = glob.glob(indir)
    print(len(file_list))
    for path in file_list:
        infile = open(path, 'r')
        outfile = open(outdir + str(file_cnt) + '.txt', 'w')
        
        lines = infile.readlines()
        line_cnt = 1
        for line in lines:
            nouns = t.nouns(line)
            buf = ""
            for noun in nouns:
                buf += noun + " "
            buf += "\n"
            outfile.write(buf)
            
            sys.stdout.write('\r| file: {0:5d}/{1:5d} | line: {2:7d}/{3:7d} | line length: {4:7d} |'.format(file_cnt, len(file_list), line_cnt, len(lines), len(line)))
            sys.stdout.flush()
            line_cnt += 1
        
        infile.close()
        outfile.close()
        file_cnt += 1

if __name__ == '__main__':
    if (sys.argv[1] == 'Kkma'):         # 메모리 오버
        t = Kkma()
        print("Kkma")
    elif (sys.argv[1] == 'Hannanum'):   # 메모리 문제 발생
        t = Hannanum()
        print("Hannanum")
    elif (sys.argv[1] == 'Komoran'):    # 메모리 오버
        t = Komoran()
        print("Komoran")
    elif (sys.argv[1] == 'Twitter'):    # 성공
        t = Twitter()
        print("Twitter")
    elif (sys.argv[1] == 'Mecab'):      # 성공
        t = Mecab()
        print("Mecab")
    # kor_tokenize(t, 'corpus/namu/*', 'corpus/train/namu_')
    # kor_tokenize(t, 'corpus/kowiki/*', 'corpus/train/kowiki_')
    # kor_tokenize(t, 'corpus/univ/*', 'corpus/train/hyu_')
    kor_tokenize(t, 'corpus/name.txt', 'corpus/vocab/'+sys.argv[1])
