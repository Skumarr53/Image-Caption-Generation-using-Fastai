from statistics import mean
from fastai.callback import Callback
import copy as cp
from torch import nn
from fastai.vision import *
from pathlib import  Path, posixpath
from pdb import set_trace
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def beam_search(mod, img,vocab = None, beam_size = 5):
    with torch.no_grad():
        k = beam_size
        
        ## imput tensor preparation
        img = img.unsqueeze(0) #treating as batch of size 1

        ## model prepartion
        #mod = learn.model

        # encoder output
        encoder_out = mod.encoder(img)
        encoder_dim = encoder_out.size(-1)
        num_pixels = encoder_out.size(1)

        # expand or repeat 'k' time
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[vocab['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = mod.decoder.init_hidden_state(encoder_out)

        references = list()
        hypotheses = list()

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            embeddings = mod.decoder.embedding(k_prev_words).squeeze(1).float()  # (s, embed_dim)
            awe, _ = mod.decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            gate = mod.decoder.sigmoid(mod.decoder.f_beta(h))
            awe = (gate * awe)

            h, c = mod.decoder.lstm(torch.cat([embeddings, awe], dim=1), (h, c))
            scores = mod.decoder.fc(h)
            scores = F.log_softmax(scores, dim=1)
            

            # Add scores to prev scores
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / len(vocab)  # (s)
            next_word_inds = top_k_words % len(vocab)  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1) stroes indices of words

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                next_word != vocab['<end>']]

            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)


            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]

    # Hypotheses
    hypotheses.append([w for w in seq if w not in {vocab['<start>'], vocab['<end>'], vocab['<pad>']}])

    return hypotheses


# Loss Function
def loss_func(input,targets, lamb=1):
    pred, decode_lengths, alphas,_ = input
    pred = pack_padded_sequence(pred, decode_lengths, batch_first=True).to(device)
    targs = pack_padded_sequence(targets, decode_lengths, batch_first=True).to(device)
    loss = nn.CrossEntropyLoss().to(device)(pred.data, targs.data.long())
    loss += (lamb*((1. - alphas.sum(dim=1)) ** 2.).mean()).to(device) #stochastic attention
    return  loss #loss(pred.data.long(), targets.data.long())



def topK_accuracy(input, targets, k=5):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    pred, decode_lengths, alphas,_ = input
    batch_size = targets.size(0)
    scores = pack_padded_sequence(pred, decode_lengths, batch_first=True).to(device)
    targ = pack_padded_sequence(targets, decode_lengths, batch_first=True).to(device)
    batch_size = targ.data.size(0)
    _, ind = scores.data.topk(k, 1, True, True)
    correct = ind.eq(targ.data.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total * (100.0 / batch_size)


class TeacherForcingCallback(Callback):
    def __init__(self, learn:Learner):
        super().__init__()
        self.learn = learn
    
    def on_batch_begin(self, epoch,**kwargs):
        self.learn.model.decoder.teacher_forcing_ratio = (10 - epoch) * 0.1 if epoch < 10 else 0
        
    def on_batch_end(self,**kwargs):
        self.learn.model.decoder.teacher_forcing_ratio = 0.

class GradientClipping(LearnerCallback):
    "Gradient clipping during training."
    def __init__(self, learn:Learner, clip:float = 0.3):
        super().__init__(learn)
        self.clip = clip

    def on_backward_end(self, **kwargs):
        "Clip the gradient before the optimizer step."
        if self.clip: nn.utils.clip_grad_norm_(self.learn.model.parameters(), self.clip)

        

class BleuMetric(Callback):
    def __init__(self,metadata = None, vocab = None):
        super().__init__()
        self.vocab = vocab
        self.metadata = metadata

    def on_epoch_begin(self, **kwargs):
        self.bleureferences = list()
        self.bleucandidates = list()

        
    def on_batch_end(self, last_output, last_target, **kwargs):
        pred, decode_lengths,_,inds = last_output
        references = self.metadata.numericalized_ref.loc[inds.tolist()]
        _,pred_words = pred.max(dim=-1)
        pred_words, decode_lengths,references = list(pred_words), decode_lengths, list(references)
        hypotheses = list()
        for i,cap in enumerate(pred_words): hypotheses.append([x for x in cap.tolist()[:decode_lengths[i]] if x not in {self.vocab['<start>'], self.vocab['<end>'], self.vocab['<pad>']}])
        #for i,cap in enumerate(pred_words): hypotheses.append([x for x in cap.tolist() if x not in {self.vocab['xxunk'], self.vocab['xxpad'], self.vocab['xxbos'], self.vocab['xxeos'],self.vocab['xxfld'],self.vocab['xxmaj'],self.vocab['xxup'],self.vocab['xxrep'],self.vocab['xxwrep']}])
        self.bleureferences.extend(references)
        self.bleucandidates.extend(hypotheses)

        

        
    def on_epoch_end(self, last_metrics, **kwargs):
        assert len(self.bleureferences) == len(self.bleucandidates)
        # print('\n'+' '.join([list(self.vocab.keys())[i-1] for i in self.bleucandidates[0]])+' | '+' '.join([list(self.vocab.keys())[i-1] for i in self.bleureferences[0][0]]))
        # print(' '.join([list(self.vocab.keys())[i-1] for i in self.bleucandidates[25]])+' | '+' '.join([list(self.vocab.keys())[i-1] for i in self.bleureferences[25][0]]))
        # print(' '.join([list(self.vocab.keys())[i-1] for i in self.bleucandidates[99]])+' | '+' '.join([list(self.vocab.keys())[i-1] for i in self.bleureferences[99][0]])+'\n')

        bleu4 = corpus_bleu(self.bleureferences, self.bleucandidates)
        return add_metrics(last_metrics,bleu4)


class BeamSearchBleu4(LearnerCallback):
    def __init__(self,learn:Learner,metadata = None, vocab = None, beam_fn = beam_search):
        super().__init__(learn)
        self.beam_fn = beam_fn
        self.vocab = vocab
        self.metadata = metadata

    def on_epoch_begin(self, **kwargs):
        self.beamreferences = list()
        self.beamcandidates = list()

    def on_batch_end(self,last_input, last_target, **kwargs):
        model_copy = cp.deepcopy(self.learn.model)
        imgs,_,_,inds = last_input
        references = self.metadata.numericalized_ref.loc[inds.tolist()]
        references = list(references)
        hypotheses = list()
        for img in imgs: hypotheses.append(self.beam_fn(model_copy,img,self.vocab)[0])
        self.beamreferences.extend(references)
        self.beamcandidates.extend(hypotheses)

    def on_epoch_end(self, last_metrics, **kwargs):
        assert len(self.beamreferences) == len(self.beamcandidates)
        print(' '.join([list(self.vocab.keys())[i-1] for i in self.beamcandidates[8]])+' | '+' '.join([list(self.vocab.keys())[i-1] for i in self.beamreferences[8][0]]))
        return add_metrics(last_metrics,corpus_bleu(self.beamreferences, self.beamcandidates))
