import numpy as np
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

def calculate_cosine_code(full_code, contents:list):


    model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")

    code_emb = model.encode(contents, convert_to_tensor=True)

    while True:

        query_emb = model.encode(full_code, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, code_emb)[0]
        top_hit = hits[0]

        print("Cossim: {:.2f}".format(top_hit['score']))
        print(contents[top_hit['corpus_id']])
        print("\n\n")
        break

    return contents[top_hit['corpus_id']]


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# embedding
def calculate_cosine(question, context):

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # content we want embeddings for
    sentences = [question, context]

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('shibing624/text2vec-base-chinese')
    model = AutoModel.from_pretrained('shibing624/text2vec-base-chinese').to(device)

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length = 512).to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    #calculate cosine similarity
    embed_question = np.array(sentence_embeddings[0].cpu())
    embed_context = np.array(sentence_embeddings[1].cpu())

    
    distances = cosine_similarity([embed_question],[embed_context])
    print("Distance", distances)

    return distances[0][0]



def max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_k, nr_candidates):
    """Calculate Max Sum Distance for extraction of keywords
    We take the nr_candidates most similar words/phrases to the document.
    Then, we take all top_k combinations from the nr_candidates and
    extract the combination that are the least similar to each other
    by cosine similarity.
    This is O(n^2) and therefore not advised if you use a large `top_k`
    Arguments:
        doc_embedding: The document embeddings
        candidate_embeddings: The embeddings of the selected candidate keywords/phrases
        candidates: The selected candidate keywords/keyphrases
        top_k: The number of keywords/keyhprases to return
        nr_candidates: The number of candidates to consider
    Returns:
         List[str]: The selected keywords/keyphrases
    """
    if nr_candidates < top_k:
        raise Exception(
            "Make sure that the number of candidates exceeds the number "
            "of keywords to return."
        )
    elif top_k > len(candidates):
        return []

    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    # Get top_k words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_k):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]


def mmr(doc_embedding, candidate_embeddings, words, top_k, diversity):
    """Calculate Maximal Marginal Relevance (MMR)
    between candidate keywords and the document.
    MMR considers the similarity of keywords/keyphrases with the
    document, along with the similarity of already selected
    keywords and keyphrases. This results in a selection of keywords
    that maximize their within diversity with respect to the document.
    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_k: The number of keywords/keyhprases to return
        diversity: How diverse the select keywords/keyphrases are.
                   Values between 0 and 1 with 0 being not diverse at all
                   and 1 being most diverse.
    Returns:
         List[str]: The selected keywords/keyphrases with their distances
    """
    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding) 
    word_similarity = cosine_similarity(candidate_embeddings) 

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]  
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_k - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

sw = '''
一
一個
一些
一何
一切
一則
一方面
一旦
一來
一樣
一種
一般
一轉眼
七
萬一
三
上
上下
下
不
不僅
不但
不光
不單
不只
不外乎
不如
不妨
不盡
不盡然
不得
不怕
不惟
不成
不拘
不料
不是
不比
不然
不特
不獨
不管
不至於
不若
不論
不過
不問
與
與其
與其說
與否
與此同時
且
且不說
且說
兩者
個
個別
中
臨
為
為了
為什麼
為何
為止
為此
為著
乃
乃至
乃至於
麼
之
之一
之所以
之類
烏乎
乎
乘
九
也
也好
也罷
了
二
二來
於
於是
於是乎
雲雲
雲爾
五
些
亦
人
人們
人家
什
什麼
什麼樣
今
介於
仍
仍舊
從
從此
從而
他
他人
他們
他們們
以
以上
以為
以便
以免
以及
以故
以期
以來
以至
以至於
以致
們
任
任何
任憑
會
似的
但
但凡
但是
何
何以
何況
何處
何時
余外
作為
你
你們
使
使得
例如
依
依據
依照
便於
俺
俺們
倘
倘使
倘或
倘然
倘若
借
借儻然
假使
假如
假若
做
像
兒
先不先
光
光是
全體
全部
八
六
兮
共
關於
關於具體地說
其
其一
其中
其二
其他
其餘
其它
其次
具體地說
具體說來
兼之
內
再
再其次
再則
再有
再者
再者說
再說
冒
衝
況且
幾
幾時
凡
凡是
憑
憑借
出於
出來
分
分別
則
則甚
別
別人
別處
別是
別的
別管
別說
到
前後
前此
前者
加之
加以
區
即
即令
即使
即便
即如
即或
即若
卻
去
又
又及
及
及其
及至
反之
反而
反過來
反過來說
受到
另
另一方面
另外
另悉
只
只當
只怕
只是
只有
只消
只要
只限
叫
叮咚
可
可以
可是
可見
各
各個
各位
各種
各自
同
同時
後
後者
向
向使
向著
嚇
嗎
否則
吧
吧噠
含
吱
呀
呃
嘔
唄
嗚
嗚呼
呢
呵
呵呵
呸
呼哧
咋
和
咚
咦
咧
咱
咱們
咳
哇
哈
哈哈
哉
哎
哎呀
哎喲
嘩
喲
哦
哩
哪
哪個
哪些
哪兒
哪天
哪年
哪怕
哪樣
哪邊
哪裡
哼
哼唷
唉
唯有
啊
啐
啥
啦
啪達
啷當
餵
喏
喔唷
嘍
嗡
嗡嗡
嗬
嗯
噯
嘎
嘎登
噓
嘛
嘻
嘿
嘿嘿
四
因
因為
因了
因此
因著
因而
固然
在
在下
在於
地
基於
處在
多
多麼
多少
大
大家
她
她們
好
如
如上
如上所述
如下
如何
如其
如同
如是
如果
如此
如若
始而
孰料
孰知
寧
寧可
寧願
寧肯
它
它們
對
對於
對待
對方
對比
將
小
爾
爾後
爾爾
尚且
就
就是
就是了
就是說
就算
就要
盡
儘管
儘管如此
豈但
己
已
已矣
巴
巴巴
年
並
並且
庶乎
庶幾
開外
開始
歸
歸齊
當
當地
當然
當著
彼
彼時
彼此
往
待
很
得
得了
怎
怎麼
怎麼辦
怎麼樣
怎奈
怎樣
總之
總的來看
總的來說
總的說來
總而言之
恰恰相反
您
惟其
慢說
我
我們
或
或則
或是
或曰
或者
截至
所
所以
所在
所幸
所有
才
才能
打
打從
把
抑或
拿
按
按照
換句話說
換言之
據
據此
接著
故
故此
故而
旁人
無
無寧
無論
既
既往
既是
既然
日
時
時候
是
是以
是的
更
曾
替
替代
最
月
有
有些
有關
有及
有時
有的
望
朝
朝著
本
本人
本地
本著
本身
來
來著
來自
來說
極了
果然
果真
某
某個
某些
某某
根據
歟
正值
正如
正巧
正是
此
此地
此處
此外
此時
此次
此間
毋寧
每
每當
比
比及
比如
比方
沒奈何
沿
沿著
漫說
點
焉
然則
然後
然而
照
照著
猶且
猶自
甚且
甚麼
甚或
甚而
甚至
甚至於
用
用來
由
由於
由是
由此
由此可見
的
的確
的話
直到
相對而言
省得
看
眨眼
著
著呢
矣
矣乎
矣哉
離
秒
稱
竟而
第
等
等到
等等
簡言之
管
類如
緊接著
縱
縱令
縱使
縱然
經
經過
結果
給
繼之
繼後
繼而
綜上所述
罷了
者
而
而且
而況
而後
而外
而已
而是
而言
能
能否
騰
自
自個兒
自從
自各兒
自後
自家
自己
自打
自身
至
至於
至今
至若
致
般的
若
若夫
若是
若果
若非
莫不然
莫如
莫若
雖
雖則
雖然
雖說
被
要
要不
要不是
要不然
要麼
要是
譬喻
譬如
讓
許多
論
設使
設或
設若
誠如
誠然
該
說
說來
請
諸
諸位
諸如
誰
誰人
誰料
誰知
賊死
賴以
趕
起
起見
趁
趁著
越是
距
跟
較
較之
邊
過
還
還是
還有
還要
這
這一來
這個
這麼
這麼些
這麼樣
這麼點兒
這些
這會兒
這兒
這就是說
這時
這樣
這次
這般
這邊
這裡
進而
連
連同
逐步
通過
遵循
遵照
那
那個
那麼
那麼些
那麼樣
那些
那會兒
那兒
那時
那樣
那般
那邊
那裡
都
鄙人
鑒於
針對
阿
除
除了
除外
除開
除此之外
除非
隨
隨後
隨時
隨著
難道說
零
非
非但
非徒
非特
非獨
靠
順
順著
首先
︿
！
＃
＄
％
＆
（
）
＊
＋
，
０
１
２
３
４
５
６
７
８
９
：
；
＜
＞
？
＠
［
］
｛
｜
｝
～
￥
i
me
my
myself
we
our
ours
ourselves
you
you're
you've
you'll
you'd
your
yours
yourself
yourselves
he
him
his
himself
she
she's
her
hers
herself
it
it's
its
itself
they
them
their
theirs
themselves
what
which
who
whom
this
that
that'll
these
those
am
is
are
was
were
be
been
being
have
has
had
having
do
does
did
doing
a
an
the
and
but
if
or
because
as
until
while
of
at
by
for
with
about
against
between
into
through
during
before
after
above
below
to
from
up
down
in
out
on
off
over
under
again
further
then
once
here
there
when
where
why
how
all
any
both
each
few
more
most
other
some
such
no
nor
not
only
own
same
so
than
too
very
s
t
can
will
just
don
don't
should
should've
now
d
ll
m
o
re
ve
y
ain
aren
aren't
couldn
couldn't
didn
didn't
doesn
doesn't
hadn
hadn't
hasn
hasn't
haven
haven't
isn
isn't
ma
mightn
mightn't
mustn
mustn't
needn
needn't
shan
shan't
shouldn
shouldn't
wasn
wasn't
weren
weren't
won
won't
wouldn
wouldn't
的
了
和
是
就
都
而
及
與
著
或
一個
沒有
我們
你們
妳們
他們
她們
是否
。
,
「
」
、
‧
《
》
〈
〉
——
—
～
【
】
［
］
（
）
：
；
？
！
︿
！
＃
＄
％
＆
（
）
＊
＋
，
０
１
２
３
４
５
６
７
８
９
：
；
＜
＞
？
＠
［
］
｛
｜
｝
～
￥
(
)
'''
def read_stopwords_list():
    stop_words = sw.split()
    return stop_words