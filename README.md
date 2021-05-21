# Pairwise Learning to Rank Approach Using LambdaMart

LambdaMART adalah sebuah evolusi algoritma perangkingan atau learning to rank (LTR). Awalnya dimulai dari RankNet yang merupakan algoritma LTR berbasis jaringan saraf atau neural network yang mengoptimalkan fungsi cost cross entropy menggunakan turunan gradient (gradient descent) dalam melatih modelnya untuk menentukan peringkat. Perkembangan selanjutnya diikuti oleh LambdaRank yang meningkatkan RankNet dengan menentukan gradient fungsi cost tertentu hanya di tempat yang ingin dituju. Menggunakan LambdaRank dapat mengoptimalkan fungsi cost sekaligus memberi keefektifitasan dalam perangkingan dan pengambilan informasi yang tidak dapat dibedakan satu dengan yang lain

LambdaMART memodelkan gradient dengan menggunakan posisi peringkat dari dokumen tersebut terkait kueri tertentu dan menggunakan gradient untuk menghitung bobot (weights) optimal untuk menggabungkan model yang lemah dalam model yang ditingkatkan. LambdaMART menggunakan pergerakan secara upward dan downward dari setiap dokumen. Pergerakan yang dihasilkan tidak sama dengan pergerakan gradient dokumen, tetapi memperlakukan aspek efektifitas setiap kueri secara setara 

Evaluasi performa dari kinerja perankingan Learning to Rank menggunakan metrik Normalized Discounted Cumulative Gain (NDCG).

Berikut adalah formula yang digunakan untuk menghitung DCG dan iDCG:
![image](https://user-images.githubusercontent.com/60861592/119132341-c6d0ca00-ba64-11eb-8a17-895c842473be.png)

Sehingga untuk mendapatkan nilai dari NDCG, dapat digunakan formula berikut ini:
![image](https://user-images.githubusercontent.com/60861592/119132426-dfd97b00-ba64-11eb-946c-1ea8263d2039.png)


### Dataset LETOR 4.0 (MQ2008)
Million queries dataset from TREC 2008 :
[MQ2008](https://onedrive.live.com/?authkey=%21ACnoZZSZVfHPJd0&id=8FEADC23D838BDA8%21107&cid=8FEADC23D838BDA8)

Untuk keterangan tiap feature dapat dilihat padaa link berikut ini:
(https://arxiv.org/ftp/arxiv/papers/1306/1306.2597.pdf)

### Hasil Evaluasi 

|Fold | NDCG |
|-----|----------------------|
|  1  | 0.574079387985382    |
|  2  | 0.5220223661651783  | 
|  3  | 0.45792178900453606    | 
|  4  | 0.5062589921310893   | 
|  5  | 0.5184832761245837 | 
|Average ndcg @10 | 0.5157531622821538 |


Rata - rata skor NDCG tertinggi terdapat pada Fold 1 yaitu : 0.574079387985382.

Sehingga rata-rata seluruh niali dari  NDCG kelima fold dijumlahkan dan akan didapat lagi rata-rata NDCG@10 sebesar 0.5157531622821538.

![image](https://user-images.githubusercontent.com/60861592/119130319-44470b00-ba62-11eb-81ff-4aa1bd80e526.png)

Rata-rata nilai NDCG yang sudah didapat dari kelima Fold memberikan nilai yang cukup baik untuk melakukan perangkingan, mengingat pada model LambdaMART tidak menggunakan framework yang dapat membantu dalam mempercepat kinerja dalam melakukan perangkingan. 



