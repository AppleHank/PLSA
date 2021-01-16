# PLSA
將LSA以機率的方式實作，透過unsupervised learning以機率歸納topic，是個非常特別的topic model，其難處有二。
第一是要如何將機率的概念寫為程式碼，例如EM algorithm中需要設置兩個變數，分別為

![image](https://github.com/AppleHank/Image/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202021-01-16%20170148.jpg?raw=true)

![image](https://github.com/AppleHank/Image/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202021-01-16%20170338.jpg?raw=true)

其中Tk是我們自己指定的topic數量，Wi是所有document的vocabulary中的第i個word，Dj為第j篇document。

P(Tk|wi,dj)其含意為固定word i,document j的情況下，topic k 的機率，也就是document j裡面的第i個字有多少機率屬於topic k，而且其底下所有topic總和應該為1。在一開始沒有接觸過類似的題型時花費了一些時間將其轉換為程式碼。

第二個難處在於運算時間，以這個作業的資料及來看，document有14995篇，vocabulary有11,000個字左右，所以假設topic設為8就有 8*11,000*14995=1,319,560,000個參數，如果沒有做任何處理，在我的筆電上運算4個topic，50個epoch的話要花費4天左右的時間。而且也無法透過矩陣運算使用GPU加速。
加速運算時間有兩種方式，一種是P(Tk|wi,dj)是document j裡面word i屬於Topic k的機率，但如果document j裡面完全沒有出現過word i的話，機率一定會是0。因此這個三維陣列中有許多元素都是0，是十分稀疏的，可以透過稀疏矩陣儲存。

第二種方式是使用numba運算，運算速度可以提升一百倍左右，但numba的限制較多，實作上稍嫌麻煩。

此程式碼僅使用numba加速，最後8個topic，50個epoch總共執行了1小時左右的時間，若加上稀疏矩陣可以加速到十分鐘以內。



