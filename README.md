# 🔥 TrustRAG: Enhancing Robustness and Trustworthiness in RAG
[[Project page]](https://trust-rag.github.io/) [[Paper]](https://arxiv.org/pdf/2501.00879)

[Huichi Zhou*](https://huichizhou.github.io/)<sup>1</sup>, [Kin-Hei Lee*](https://openreview.net/profile?id=~Lee_KinHei1)<sup>1</sup>, [Zhonghao Zhan*](https://zhonghaozhan.github.io/)<sup>1</sup>, [Yue Chen]()<sup>2</sup>, [Zhenhao Li](https://zhenhaoli.net/)<sup>1</sup>, [Zhaoyang Wang](https://zhaoyang.win/)<sup>3</sup>,[Hamed Haddadi](https://profiles.imperial.ac.uk/h.haddadi)<sup>1</sup> [Emine Yilmaz](https://scholar.google.com/citations?hl=en&user=ocmAN4YAAAAJ)<sup>4</sup>

<sup>1</sup>Imperial College London, <sup>2</sup>Peking University, <sup>3</sup>University of North Carolina at Chapel Hill, <sup>4</sup>University College London

<sup>*</sup>Equal Contribution

<img src="media/Method.jpg" alt="drawing" width="100%"/>

## 🔥 NEWS

- 2025.1.10 OpenAI API Inference Now Supported! Additionally, we have introduced a new module: Self-Assessment of Retrieval Correctness, enabling enhanced evaluation of retrieval accuracy.

## 🛝 Try it out!

### 🛠️ Installation

```
git clone https://github.com/HuichiZhou/TrustRAG.git

conda create -n trustrag python=3.10

conda activate trustrag

pip install lmdeploy

pip install beir

pip install nltk

pip install rouge_score

pip install timm==0.9.2

cd TrustRAG

python run.py
```

Optional: [RAGDefender](https://github.com/SecAI-Lab/RAGDefender) (ACSAC 2025) can be used as the passage removal step instead of k-means (`pip install ragdefender`, then `--removal_method ragdefender` or `'removal_method': 'ragdefender'` in **run.py**).

## 🙏 Acknowledgement

* Our code used the implementation of [corpus-poisoning](https://github.com/princeton-nlp/corpus-poisoning).
* The model part of our code is from [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection).
* Our code used [beir](https://github.com/beir-cellar/beir) benchmark.
* Our code used [contriever](https://github.com/facebookresearch/contriever) for retrieval augmented generation (RAG).
* Our code used [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) for corpus poisoning attack.


## 📝 Citation and Reference

If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:

```
@article{zhou2025trustrag,
  title={Trustrag: Enhancing robustness and trustworthiness in rag},
  author={Zhou, Huichi and Lee, Kin-Hei and Zhan, Zhonghao and Chen, Yue and Li, Zhenhao and Wang, Zhaoyang and Haddadi, Hamed and Yilmaz, Emine},
  journal={arXiv preprint arXiv:2501.00879},
  year={2025}
}
```
