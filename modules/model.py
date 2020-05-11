from torch import nn
from torch.nn import functional as F, init
from torchvision import transforms, models
import torch
import random
from pdb import set_trace


device =torch.device("cuda" if torch.cuda.is_available() else "cpu")



# create a embedding layer
def create_emb(embedding_array):
    emb = nn.Embedding(len(word_map),embedding_dim)
    emb.weight.data = torch.from_numpy(embedding_array).float()
    return emb

class Encoder(nn.Module):
    def __init__(self,encode_img_size, fine_tune = False):
        super(Encoder, self).__init__()
        self.enc_imgsize = encode_img_size
        resnet = models.resnet101(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2]) # removing final Linear layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encode_img_size,encode_img_size))
        self.fine_tune = fine_tune
        self.fine_tune_h()
        
    def fine_tune_h(self):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.encoder.children())[5:]:
            for p in c.parameters():
                p.requires_grad = self.fine_tune
        
    def forward(self,X):
        out = self.encoder(X) # X is tensor of size (batch size, 3 (RGB), input height, width)
        out = self.adaptive_pool(out) # output (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)
        out = out.view(out.size(0), -1, out.size(3))
        return out
    
class Decoder(nn.Module):
    def __init__(self,attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5, pretrained_embedding = None,teacher_forcing_ratio = 0):
        super(Decoder, self).__init__()
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim) 
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True) #use 
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # gate
        self.pretrained_embedding = pretrained_embedding
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()
        
    def init_weights(self):
        """
        Initilizes some parametes with values from the uniform Dist

        """
        self.embedding.weight.data.uniform_(0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1,0.1)

        # Kaiming initialization
        #init.kaiming_normal_(self.init_h.weight, mode='fan_in')
        #init.kaiming_normal_(self.init_c.weight, mode='fan_in')
        #init.kaiming_normal_(self.f_beta.weight, mode='fan_in')
        #init.kaiming_normal_(self.fc.weight, mode='fan_in')

    def pretrained(self):
        if self.pretrained_embedding is not None:
            self.embedding.weight.data = torch.from_numpy(self.pretrained_embedding)
            
    def init_hidden_state(self, encoder_out):
        
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c
            
    def forward(self,encoder_out, encoded_captions,decode_lengths,inds):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        num_pixels = encoder_out.size(1)
        #embeddings = self.embedding(encoded_captions)
        
        ## initililize hidden encoding
        h, c = self.init_hidden_state(encoder_out)
        
        #dec_out = torch.zeros(1,batch_size,self.decoder_dim).to(device) #uncomment for teacher forcing

        decode_lengths = decode_lengths - 1
        
        max_len = max(decode_lengths).item()
        
        
        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max_len, vocab_size)
        alphas = torch.zeros(batch_size, max_len, num_pixels)
        
        for t in range(max_len):
            batch_size_t = sum([l.item() > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            # teacher forcing 
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            
            
            inp_emb = self.embedding(encoded_captions[:batch_size_t,t]).float() if  (use_teacher_forcing or t==0) else self.embedding(prev_word[:batch_size_t]).float()
            #self.emb2dec_dim((embeddings[:batch_size_t, t, :]).float()) use syntax for teacher forcing
            #inp_emb = inp_emb if (use_teacher_forcing or t==0) else dec_out.squeeze(0)[:batch_size_t] #uncomment to add teacher forcing
            
            h, c = self.lstm(
                torch.cat([inp_emb, attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t,t , :] = preds
            alphas[:batch_size_t, t, :] = alpha

            _,prev_word = preds.max(dim=-1)
        return predictions,decode_lengths, alphas, inds
        
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        
        self.enc_att = nn.Linear(encoder_dim,attention_dim)
        self.dec_att = nn.Linear(decoder_dim,attention_dim)
        self.att = nn.Linear(attention_dim,1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # kaiming
        #init.kaiming_normal_(self.enc_att.weight, mode='fan_in')
        #init.kaiming_normal_(self.dec_att.weight, mode='fan_in')
        #init.kaiming_normal_(self.att.weight, mode='fan_in')

    def forward(self,encoder_out, decoder_hidden):
        encoder_att = self.enc_att(encoder_out)
        decoder_att = self.dec_att(decoder_hidden)
        att = self.att(self.relu(encoder_att + decoder_att.unsqueeze(1))).squeeze(2) #testing added batchnorm 
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out*alpha.unsqueeze(2)).sum(dim=1)
        
        return attention_weighted_encoding, alpha