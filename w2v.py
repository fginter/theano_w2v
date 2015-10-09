import sys
import argparse
import numpy
import index
import net
import codecs
import random
import time

def read_conll(inp,maxsent):
    """ Read conll format file and yield one sentence at a time as a list of lists of columns. If inp is a string it will be interpreted as filename, otherwise as open file for reading in unicode"""
    if isinstance(inp,basestring):
        f=codecs.open(inp,u"rt",u"utf-8")
    else:
        f=codecs.getreader("utf-8")(sys.stdin) # read stdin
    count=0
    sent=[]
    comments=[]
    for line in f:
        line=line.strip()
        if not line:
            if sent:
                count+=1
                yield sent, comments
                if maxsent!=0 and count>=maxsent:
                    break
                sent=[]
                comments=[]
        elif line.startswith(u"#"):
            if sent:
                raise ValueError("Missing newline after sentence")
            comments.append(line)
            continue
        else:
            sent.append(line.split(u"\t"))
    else:
        if sent:
            yield sent, comments

    if isinstance(inp,basestring):
        f.close() #Close it if you opened it



def minibatches(inp,inp_vocab,outp_vocab,minibatch_focus,minibatch_context,win_max_len):
    #minibatch_* is a vector of integers 
    minibatch_size=len(minibatch_focus)
    sent_counter=0
    idx=0
    for sent,comments in read_conll(inp,0):
        sent_counter+=1
        if sent_counter%1000==0:
            print >> sys.stderr, "\n\nAT SENTENCE ",sent_counter,"\n\n"

        for focus_idx in range(len(sent)):
            focus=sent[focus_idx][1]
            focus_rank=inp_vocab.vocab.get(focus)
            if focus_rank is None:
                continue
            window=random.randint(1,win_max_len)
            for context_idx in range(max(0,focus_idx-window),min(len(sent),focus_idx+window)):
                if context_idx==focus_idx:
                    continue
                context=sent[context_idx][1]
                context_rank=outp_vocab.vocab.get(context)
                if context_rank is None:
                    continue
                minibatch_focus[idx]=focus_rank
                minibatch_context[idx]=context_rank
                idx+=1
                if idx==minibatch_size:
                    yield  True
                    idx=0


def train_w2v(args):
    minibatch_size=1000
    win_max_len=7
    minibatch_focus=numpy.zeros((minibatch_size,),numpy.int32)
    minibatch_context=numpy.zeros((minibatch_size,),numpy.int32)

    inp_vocab=index.Vocab.load(args.vocab,1000000)
    outp_vocab=index.Vocab.load(args.vocab,5000)
    sg=net.SkipGram.empty(inp_vocab,outp_vocab,100)

    for _ in minibatches(sys.stdin,inp_vocab,outp_vocab,minibatch_focus,minibatch_context,win_max_len):
        sg.train(minibatch_focus,minibatch_context,0.001)
        #sg.outputf(minibatch_focus)
        #print time.ctime()
        #print minibatch_focus
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train w2v')
    parser.add_argument('--vocab', default="vocabulary_1M.pkl", help='Vocabulary file produced by index.py. Default: %(default)s.')
    args = parser.parse_args()
  
    train_w2v(args)
