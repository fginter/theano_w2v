import sys
import codecs
import argparse
import cPickle as pickle

class Vocab(object):

    def __init__(self,vocab):
        self.vocab=vocab #key: unicode  value: (idx,count)

    @staticmethod
    def read_conllu(src,max_lines):
        """src should produce lines with conll-data, open file is enough"""
        vocab={}
        seen_words=0 #How many words of text have I seen?
        for line in src:
            line=line.rstrip(u"\n")
            if not line or line[0]==u"#":
                continue
            cols=line.split(u"\t",2)
            if not cols[0].isdigit():
                continue
            vocab[cols[1]]=vocab.get(cols[1],0)+1
            seen_words+=1
            if max_lines>0 and seen_words==max_lines:
                break
            if seen_words%100000==0:
                print >> sys.stderr, "...",seen_words, "words seen"
        return vocab

    @staticmethod
    def cut_vocab(vocab,max_rank):
        top_n=sorted(vocab.iteritems(),key=lambda (w,c): c,reverse=True)[:max_rank] #top_n keys, sorted
        items=((w,idx) for idx,w in enumerate(top_n))
        return dict(items) #Dictionary with top_n words, and their indices

    def save(self,f_name):
        with open(f_name,"wb") as f:
            pickle.dump(self.vocab,f,protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls,f_name):
        with open(f_name,"rb") as f:
            vocab=pickle.load(f)
        return cls(vocab)

    @classmethod
    def from_conllu(cls,src,args):
        vocab=cls.cut_vocab(cls.read_conllu(src,args.max_lines),args.max_rank)
        return cls(vocab)
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Index conll on input. Save vocabulary. Try -h')
    parser.add_argument('-o', '--output', default="vocabulary.pkl", help='Name of the output vocabulary name.')
    parser.add_argument('--max-rank', default=500000, type=int, help="Only keep top N words. Default: %(default)d")
    parser.add_argument('--max-lines', default=1000000, type=int, help="Only index the first N lines from the input. Default: %(default)d")
    args = parser.parse_args()
    v=Vocab.from_conllu(codecs.getreader("utf-8")(sys.stdin),args)
    v.save(args.output)
