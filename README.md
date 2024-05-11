# Bayesian Regression with Pyro

Этот Python-код демонстрирует байесовскую регрессию с использованием библиотеки Pyro. Байесовская регрессия - это мощная техника для моделирования взаимосвязей между переменными с учетом неопределенности параметров модели. Код охватывает генерацию данных, определение байесовской модели регрессии, проведение байесовского вывода с использованием стохастической вариационной инференции (SVI) и визуализацию апостериорных распределений параметров модели.

## **Теория баесовской линейной регрессии**

### **Введение**

---

Линейная регрессия - популярный регрессионный подход в машинном обучении. Линейная регрессия основана на предположении, что базовые данные распределены нормально и что все соответствующие переменные-предсказатели имеют линейную зависимость от результата. Но в реальном мире это не всегда возможно, это будет следовать этим предположениям, байесовская регрессия могла бы быть лучшим выбором.
Байесовская регрессия использует предшествующие убеждения или знания о данных, чтобы “узнать” о них больше и создать более точные прогнозы. Она также учитывает неопределенность данных и использует предшествующие знания для предоставления более точных оценок данных. В результате это идеальный выбор, когда данные сложны или неоднозначны.

Байесовская регрессия использует алгоритм Байеса для оценки параметров модели линейной регрессии на основе данных, включая предварительные знания об этих параметрах. Из-за своего вероятностного характера она может давать более точные оценки параметров регрессии, чем обычная линейная регрессия методом наименьших квадратов (OLS), обеспечивать меру неопределенности в оценке и делать более надежные выводы, чем OLS. Байесовская регрессия также может быть использована для связанных задач регрессионного анализа, таких как выбор модели и обнаружение выбросов.


### **Байесовская регрессия**

---

Байесовская регрессия - это тип линейной регрессии, которая использует байесовскую статистику для оценки неизвестных параметров модели. Она использует теорему Байеса для оценки вероятности набора параметров с учетом наблюдаемых данных. Целью байесовской регрессии является нахождение наилучшей оценки параметров линейной модели, которая описывает взаимосвязь между независимыми и зависимыми переменными.

Основное различие между традиционной линейной регрессией и байесовской регрессией заключается в лежащем в основе предположении относительно процесса генерации данных. Традиционная линейная регрессия предполагает, что данные соответствуют гауссовскому или нормальному распределению, в то время как байесовская регрессия имеет более сильные предположения о характере данных и устанавливает априорное распределение вероятностей для параметров. Байесовская регрессия также обеспечивает большую гибкость, поскольку допускает дополнительные параметры или предварительные распределения, и может использоваться для построения сколь угодно сложной модели, которая явно выражает предыдущие представления о данных. Кроме того, байесовская регрессия обеспечивает более точные прогностические показатели по меньшему количеству точек данных и способна строить оценки с учетом неопределенности вокруг оценок. С другой стороны, традиционные линейные регрессии проще реализовать и, как правило, быстрее с более простыми моделями и могут обеспечить хорошие результаты, когда предположения о данных верны.

Байесовская регрессия может быть очень полезна, когда у нас недостаточно данных в наборе данных или данные плохо распределены. Выходные данные байесовской регрессионной модели получаются из распределения вероятностей, по сравнению с обычными методами регрессии, где выходные данные получаются только из одного значения каждого атрибута.

### **Некоторые зависимые концепции для байесовской регрессии**

---

**Теорема Байеса**

Теорема Байеса дает соотношение между предшествующей вероятностью события и его последующей вероятностью после учета доказательств. В ней говорится, что условная вероятность события равна вероятности события при определенных условиях, умноженной на предшествующую вероятность события, деленную на вероятность условий.

т.е. P(A | B) = $\frac{P(B | A) \cdot P(A)} {P(B)}$.

где P (A | B) - вероятность наступления события A, учитывая, что событие B уже произошло, P (B | A) - вероятность наступления события B, учитывая, что событие A уже произошло, P (A) - вероятность наступления события A и P (B) - вероятность наступления события B.


**Оценка максимального правдоподобия (MLE)**

MLE - это метод, используемый для оценки параметров статистической модели путем максимизации функции правдоподобия. он стремится найти значения параметров, которые делают наблюдаемые данные наиболее вероятными в рамках предполагаемой модели. MLE не включает никакой предварительной информации или предположений о параметрах и предоставляет точечные оценки параметров.

**Максимальная апостериорная оценка (MAP)**

Оценка MAP - это байесовский подход, который объединяет предварительную информацию с функцией правдоподобия для оценки параметров. Он включает в себя нахождение значений параметров, которые максимизируют апостериорное распределение, которое получается путем применения теоремы Байеса. При оценке карты для параметров задается априорное распределение, представляющее предшествующие убеждения или знания об их значениях. Затем функция правдоподобия умножается на предыдущее распределение, чтобы получить совместное распределение, и значения параметров, которые максимизируют это совместное распределение, выбираются в качестве оценок карты. Оценка MAP предоставляет точечные оценки параметров, аналогичные MLE, но включает в себя предварительную информацию.

### **Необходимость в байесовской регрессии**

---

Существует несколько причин, по которым байесовская регрессия полезнее других методов регрессии. Некоторые из них следующие:

1. Байесовская регрессия также использует исходное представление о параметрах в анализе. что делает ее полезной, когда доступны ограниченные данные и предварительные знания являются релевантными. Объединяя предварительные знания с наблюдаемыми данными, байесовская регрессия обеспечивает более информированные и потенциально более точные оценки параметров регрессии.
2. Байесовская регрессия обеспечивает естественный способ измерения неопределенности в оценке параметров регрессии путем генерации апостериорного распределения, которое отражает неопределенность в значениях параметров, в отличие от одноточечной оценки, которая производится стандартными методами регрессии. Это распределение предлагает диапазон приемлемых значений параметров и может использоваться для вычисления достоверных интервалов или байесовских доверительных интервалов.
3. Для включения сложных корреляций и нелинейностей байесовская регрессия обеспечивает гибкость, предлагая структуру для интеграции различных предыдущих распределений, что делает ее способной обрабатывать ситуации, когда основные предположения стандартных методов регрессии, такие как линейность или гомоскедастичность, могут быть неверными. Это позволяет моделировать более реалистичные и детализированные взаимосвязи между предикторами и переменной отклика.
4. Байесовская регрессия облегчает выбор и сравнение моделей путем вычисления апостериорных вероятностей различных моделей.
5. Байесовская регрессия может обрабатывать выбросы и влиятельные наблюдения более эффективно по сравнению с классическими методами регрессии. Она обеспечивает более надежный подход к регрессионному анализу, поскольку экстремальные или влиятельные наблюдения оказывают меньшее влияние на оценку.

### **Реализация байесовской регрессии**

---

Пусть независимыми объектами для линейной регрессии будут $X = \{x_1, x_2, ..., x_P\}$,
где xᵢ представляет i-е-е независимые объекты, а целевыми переменными будут Y.

Предположим, у нас есть n выборок (X, y). Линейная зависимость между зависимой переменной Y и независимыми признаками X может быть представлена в виде:

$y = w_0 + w_1x_1 + w_2x_2 + ... + w_px_p + \epsilon$

или

$y = f(x,w) + \epsilon$

Здесь $w = \{w_0, w_1, w_2, ..., w_p\}$  приведены коэффициенты регрессии, представляющие взаимосвязь между независимыми переменными и зависимой переменной, а ε - член ошибки.

Мы предполагаем, что ошибки (ε) следуют нормальному распределению со средним значением 0 и постоянной дисперсией, $\sigma^2$  т. е. $(\epsilon \sim N(0, \sigma^2))$ . Это предположение позволяет нам моделировать распределение целевой переменной вокруг прогнозируемых значений.

**Функция правдоподобия**

Распределение вероятностей, которое выстраивает взаимосвязь между независимыми признаками и коэффициентами регрессии, известно как сходство. Оно описывает вероятность получения определенного результата при некоторой комбинации коэффициентов регрессии.

**Допущения:**

Ошибки $\epsilon = \{\epsilon_1, \epsilon_2, ..., \epsilon_p\}$ независимы и идентичны и следуют нормальному распределению со средним значением 0 и дисперсией $\sigma^2$.
Это означает, что целевая переменная Y, учитывая предиктор ${X_1, X_2, ..., X_p}$, следует нормальному распределению со средним ${\mu = f(x,w) = w_0 + w_1x_1 + w_2x_2 + ... + w_px_p}$  значением и дисперсией $\sigma^2$ .
Следовательно, функция плотности условной вероятности PDF для Y с учетом переменных-предикторов будет равна:

![image.png](https://github.com/denis-samatov/Bayesian_regression_model/blob/main/formula_1.png)
Функция правдоподобия для n наблюдений, у которых каждое наблюдение $(x_{i1}, x_{i2}, \cdots, x_{iP}, y_i)$  соответствует нормальному распределению со средним ${\mu_i = w_{0} + w_1x_{i1} + w_2x_{i2} + \cdots + w_{P}x_{iP}}$ и дисперсией $\sigma^2$ , будет совместной функцией плотности вероятности PDF зависимых переменных и может быть записана как произведение отдельных PDF:

![image.png](https://github.com/denis-samatov/Bayesian_regression_model/blob/main/formula_2.png)
Для упрощения вычислений мы можем взять логарифм функции правдоподобия:

![image.png](https://github.com/denis-samatov/Bayesian_regression_model/blob/main/formula_3.png)
**Точность:**

${\beta = \frac{1}{\sigma^2}}$

Используя условное выражение PDF из предыдущего ответа, мы подставляем его в функцию правдоподобия:

${\ln(L(y | x, w,\sigma ^ 2))= \frac{N}{2}\ln(2\pi)-\frac{N}{2}\ln(\beta)-\frac{\beta}{2}\sum_{i=1}^{N}\left[(y-f(x_i,w))^2 \right ]}$

Вероятность отрицательного логарифма:

${-\ln(L(y | x, w,\sigma ^ 2))= \frac{\beta}{2}\sum_{i=1}^{N}\left[(y-f(x_i,w))^2 \right]-\frac{N}{2}\ln(2\pi)+\frac{N}{2}\ln(\beta)}$

Здесь $\ln(2\pi)$  и $\ln(\beta)4  являются постоянными, поэтому,

${-\ln(L(y|x, w,\sigma ^ 2))= \frac{\beta}{2}\sum_{i=1}^{N}\left[(y-f(x_i,w))^2 \right ] + \text{constant}}$

**Prior:**

Prior - это первоначальное убеждение или вероятность относительно параметра до наблюдения за данными. Это информация или предположение о параметрах.
При максимальной апостериорной оценке (MAP) мы включаем предварительные знания или убеждения о параметрах в процесс оценки. Мы представляем эту предварительную информацию, используя предыдущее распределение, обозначаемое ${P (w|\alpha) =N (0,\alpha^{-1}I)}$

**Апостериорное распределение:**

Апостериорное распределение - это обновленные убеждения или распределение вероятностей параметров, которое получается после учета предыдущего распределения параметров и наблюдаемых данных.
Используя теорему Байеса, мы можем выразить апостериорное распределение в терминах функции правдоподобия и предшествующего распределения:

${P(w |X,\alpha, \beta^{-1}) = \frac{L(Y|X,w, \beta^{-1}) \cdot P(w|\alpha)}{P(Y|X)}}$

P (Y|X) - предельная вероятность наблюдаемых данных, которая действует как нормализующая константа. Поскольку она не зависит от значений параметров, мы можем игнорировать ее в процессе оптимизации.

${P(w | X,\alpha, \beta^{-1}) \propto(L(Y | X,w, \beta^{-1}) \cdot P(w|\alpha))}$

В приведенной выше формуле,
Логарифмическое правдоподобие: ${L(Y|X, w, \beta^{-1})}$ является нормальным распределением,
Prior: ${P(w|\alpha)}$ является однородным.

Таким образом, апостериорное распределение также будет нормальным.

На практике часто удобнее работать с логарифмом апостериорного распределения, известным как логарифмически-апостериорный:
Для получения максимального апостериорного распределения мы используем отрицательное правдоподобие:

[image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZwAAAClCAYAAABoQ0voAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAEXQSURBVHhe7Z0JuFVT+8CXKZGMyZAUEdGgyFCRKRlKhkQRKZGhkNDgQ6VBZWpCiaQ/JVOZM2TMkAaSoaTMZPjSJzP7v37vXevefffd55x97j3jPev3POe5Z69z7jn77L3Weof1vu/aYLsaO3jK4XA4HI40s6H563A4HA5HWhEL5713F5tDh8NRmbj33qlq2223Ve3bn2haHI7sIQJn5SfL1VZbbWmaHA5HPvPQQw+pO+64Q73wwgtqgw02UMOHD1f9+/c3rzoc2cO51ByOSkbHjh3V888/rzzPk8e///5rXnE4sosTOA6Hw+HICE7gVIDPP/9cXBb2MWXKFPNKWdA0L7nkEnFzZILZs2erK664whylj1WrVqmDDz7YHDnyDfrIW2+9ZY4qD9ddd536v//7P3OUHnbZZRf1+++/m6PUcfbZZ6tPP/3UHMWme/fu6ttvvzVH4Wy00UbF89N//vMf05o9yrWG89NPP6mvv/5aNWzY0LTkLh999JF6//331X//+195XHzxxapatWrm1YrxxRdfqCZNmqhHHnlEjvfee2+14447yvMgPXr0UFtvvbW66aabTEtZ3nnnHfXLL7/I81q1aqk///xTff/993IMrVu3lo6TiCeffFL169dPLVy4UG2++ebS9u6778rvh3333Vd98MEHIgQh1ud+/PHH6ptvvlE1a9aUwcVnbLfddmqfffYx71DqvffeU506dZLrHI9FixapdevWyffsv//+6rPPPpPftummm6pDDjnEvEupH3/8US1dulRtsskm0r8459WrV6s99thDzqEy8Nxzz8lkyGL+P//8o3r27Jm2scT1HjZsmBo4cKBpKc0PP/wg9591nwYNGpjWzPHJJ5+oL7/80hyFU6VKFRkP3H8m0EQwsX744Yfym9LJlltuKYKhRo0apiU1MO5OOeUU9dJLL6ntt9/etJaFMdWtWzf16quv6vl7K9NampdfflnG+T333KPq1q2rBg8ebF7JEgictWt/1ucUjbVr13qNGjXy9IXwli9fblpzFz3Be82aNWNmlceaNWvMKxVHT5reDjvsYI5iM2rUKE8LJu+PP/4wLeH07dvX04NfPtOeb/369aWNh558zTtjoydrb5tttvEWL15sWorQE46nLZHiz7Wf3blzZ08LNvOu0nDeWsAUv19PkN61115rXi1CCyFvr732Mkexufzyy72DDjqo1PfzaNWqlXlHEfPmzfNq165d6j077bSTN2PGDPOO/OXXX3/1zjnnHPkt//77r7Rpxc3TwsabP3++HKcarp8WOOYoHD2xefXq1ZOxnWn0RCh9Wytrxfd7v/32K+7z9A/6M+1aYfOmTJli/jOcadOmeVoweT//HH1OKy/Vq1f3tNJkjlIL14Xf/vfff5uWcG677Tavbdu25ig2Wgh711xzjTnKHkkJHDokkzcn//DDD3t16tTxtIQ3r+Y2xxxzjHTa7777zrRUnCgCR1tBntbQPG0BmJbEdOnSRc71tNNOMy3RadmypTd06FBzVBYEsB3YWrs1rbFBUOy8885yv8OIKnAs/u/X2q1pLc2yZcs8rZlLP0vmuuU65513XujvGT58uHf44YebI08mGQRv1MeCBQvMf5aF65xI4EDv3r3lXmeLSZMmybn6r4MfbeV6HTp0kPdwvcJAyKAUcU0yQToFDiBIJkyYYI5ic8ghh3jaYjZH4eSKwIm8hoOr59hjj5XwyiFDhojJN3PmTHXqqaeKaynX0cLRPMssgwYNkqghPSmblsS8+OKL8vfcc8+Vv1F56qmnxMV16aWXmpay6ElPVa1aVZ7PmjVL/sYC1xaft2TJErnfqeDCCy8sNv9ZZwqCT7xXr17qvvvuk36WzHXLZe6//36lrZvQ36M1ePXaa68VR5NpS1jcKVEfb7/9tvxfRbjqqqvUxIkTxdWZDXD9wIknhucLMX45P2BMacVRnvsZOXKkuCa10DIt+Q33RAsJcUXH48orr5QH/SbniWrhoD289dZb5qgENI8VK1aYo9ylZ8+eoh1l0sL55ptvvA033FDcXFFB6+c8+b///e9/pjUauCAGDBhgjmJz+umny3cceOCBpqUsX375pacVDE8rGqYlnGQtHOjTp498P66Pf/75x7R63l9//SWW6OjRo01L5QCX5fnnn2+OynLJJZd4NWrUMEephescxcIBrRh5F1xwgTnKLLVq1ZJzXbRokWkpy/r16+U9PB577DHTWgSvbbnllt6TTz5pWtJPui0cwL2YaDzgnt19993FSoxF3lk4LJDpCcoc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556r9tprL9NSAsUyX3nlFbFGUsG3336rpk6dqvbYYw/Jrbn++uvVL7/8Yl4tYdttt5X3ufwbh8NRmSjoWmpYGX369CmTbGnBpUak2DfffGNaHA6Hw1FeCtrCId+lXbt25qg03333nXrkkUfE+nE4HA5HxSlogUNU2rHHHmuOSsPCPW62AQMGmBaHw+FwVISCdanhTuvVq5cUzHzqqackKmzrrbdWH330kfr111/V77//ru6//35Vo0YN8x8OhyPXoSzUo48+KmkHtWrVEpc4gUC33nqr2nXXXc27HNmiYC0cwp3t7pnbb7+9+vHHH+U5wgZ3GlFi2223nbQ5HI7cByFTp04dETIjRoyQXDqED3UKjznmGBnXlpdfflne78gwWDhr1/7sFRpDhw71VqxYYY5Ks2bNGm+rrbbypk2bZlocDkcus3z5cm+LLbbwjj/+eNNSglYivc0339zr27evafG88847z/v558Kb97JNwVo4H3zwgYQnh4HFc9hhh8WMXnM4HLkFEaXr169XEyZMMC0l4K1o2LChuM6BbdvxZGy5pdvlONMU5BoO6zcXXHCBuvvuu01LWUj6fPzxx6VzRgEXHcmj2YCK0mxr4HAUIg8++KAkbbdv317NmTPHtJaGaFQEDlU+KBtFsJCr5JF5ClLgkMypTXClzWrTUpbjjjtOLV68WJI1o8DnBZNHTzrppAptL8D2BRtttJHkARHMwHnHYsWKFTEtNoejMsMaDcVw7733XnX22Web1tKcccYZaubMmVIY95ZbbsmI9wIratSoUeYoGmxR4t+ypNJRiGs4rN/oCdwcleXff//1tt9+e++oo44yLdEYO3aspy9p8WP33Xf3fvnlF/NqxdHWljd37lxv5MiRXuvWrUt912WXXWbe5XAUFptttpmMgdWrV5uWsmhBJO8566yzYq7dphrWgJs1a+YtXLjQtHjeX3/95W2wwQbeqaeealqKeOKJJ7wqVap4Wsk0LZWTghQ4WiPy/vzzT3NUlieffFI653333WdaonP00UcXCwHbwdPFBx984B1wwAHyPVtuuaW3fv1684rDURigHG666aYS5BOPc845R8bJ3XffbVrSD0Llq6++MkdFvP7663Iet912m2kp4YQTTjDPKi8FFzRAfg0hkbHK1ehrorQFpLR1o7SwMK3ReeCBB9SOO+5ojoqqGdCWDho0aCChnVdccYVat26d1F9zOAoJcmzq16+vNt98c9MSjlYw5S/ut0zwww8/qN12203tvPPOpqWIF198Uf6GFeatWbOmeVZ5KTiB8+abb4ogYaExjMsuu0w6Z6zXE0GiqN0i2kKAwqpVq8xR6hkzZoxsqRA1cIDBQJUFR3RQUOKtoeUD9H1/Lkos+J1s1ZEvnH/++bLWanPp/LAOylYjn3zyiRyjcAIBQemENVXOKwg5QswRjRo1Mi1FoOgWQnX4ghM43HA6INx1113FnXTRokUSKLDFFlvIwKRic3nBOurXr585Uup///ufLFqmE0rx0GETCRImTs6vdu3apqUwYZGZat0TJ06USSken376qUQ17b777qYlP9l3331la4xEQofJkBqC//3vf01LbsN2HhdeeKGMsffee0/aGNdUgke5PPnkk0WBJAwaYYqVwf1PJySV77nnnuaoCKogvPbaa6pVq1ampQQsNSJjKz2FtoZz2mmnmWeepycSb8KECd51113naQvB05OxeSU1NG3atHgth8eAAQPMK+lBCzZPa27mqCwEMDRu3NibNGmSaSlM7rjjDq9u3brerFmzJNiiZ8+e5pWyaE3f0xOHrOtVBt5//33vsMMO87Smb1rCWbx4sde2bVtPT5KmJffRgsS75JJLPC1gvOHDh3ta+ZI1Hgtrnl26dPHuuece05JZXn75ZZkHbr31VtNSeBSUwGGQde7c2RylHwRatWrVigUO0Sl0umzRrVs375RTTjFHhYnWfCUa6P7775d7sdFGG4nwiUWHDh08rT2bo8rB+PHjZWJOxLBhw7xBgwaZI0dF0Ra1zAMI80KloAQOGhAWTSaZMmVKscDhseOOO4rWnGk+/vhjmWgp21PIoF1yH7BmX3jhBbkfvXv3Nq+Whoii6tWre+vWrTMtlQMUL373/PnzTUs4/O6tt97aW7p0qWlJDUuWLDHPCgtSGSi/U9lDn+NRUALn2muvFZdCpiE80i90shH+iGXTo0cPc1S4tGnTxqtZs6Y5ik+rVq28/v37m6PKxZAhQ7yDDz7YHMXmyiuv9I455hhzlBqw9Att0kXIb7zxxt6JJ55oWgqTghI4uEeyAZpi7dq1SwmdiRMnmlfTD7kAuI6wcgoZcq/I2Tj88MNNS2zee+89mSAqq0X49ddfSz9kXSMeJEjzvs8++8y0VAzWVPi8eHlwlRGsaX43a8WFTEFFqSVbZiJVVK9eXc2aNUttuGHJ5e7bt6+Uw8kE5OewFwj5CoUMBVuppdW4cWPTEhv2SWrWrJkUcq2M7LTTTlIK6b777jMt4VCuiTBe9oZylB8bFVcIoc/xKCiBk80J96CDDlL/+c9/zFFRPsCpp55anJCWTkh0bd68uTkKh5BNNqGj2q7Wfk1rabS2r3777TdzlD8Q8k44/OzZs+WY+nQc89Aat7QFIXw20TWzLFmyRApCkgdFCDxwX1EyCL2uaAgufeWxxx4TIRirmOzKlSvNsyIIaea749UCbNmyZaR8rEMPPVQ9//zz5qjyQ58gdHrcuHHq888/N62lIVReK+zmqCjfB4X23XffNS1KnnMPSMy2OXL0E9oo9huFr776SsYkYzMsfJ/xGAxzf/XVV+Xcc5JCC4vOJvitteAR09o+okQLVQRcGCxUJjLlu3bt6j377LNSq40yOXrCMq8UoScu2VOkT58+piV/oIwIoe+sWdhrzjEP9koJogexuCAT7YeEX54w+1tuuUWPobUSbtuiRQsJCqEdN8ozzzzjVa1a1Xv66afNfyUH39GxY0cJYKAsS506daQelx+izvhdX3zxhWnxpKQSbVrJMS1luf3222U9Jewa+LnxxhulXhl9qaLkg0vtggsu8B5++GFvwYIFcu9wP/qx4c1PPfWUafGkviFt/jUa7gf3LwglqKLsxcP3E1mqBYqsPV599dXmlRK00iBh7ha+z9aWS+QuzQZZEzjEyhO1Ud5H+/btzSflF1pjksgnOoR9ULgvXdDp+Y5HHnnEtJSFvBQ7IZKbwvvJYfBDXTnamaTyFfrMhhtu6P3xxx+mJRxtEclv1ZqiaQmnX79+nraEzFHRRl/8H0EJds3jwAMPlLa77rpLjpPlmmuukfMBrR3LZ3344YdybKE2YDAQggg01qsoWhmLRx99VD4vUSDNvffeK+/zC7TykusCh/5vayhqq07OdebMmXJsIXeLdgSBRVuYsk5L7l0qQOnp1KlTsXJBzcT99ttPnltQAjmPyy+/3LQUQaAH7SiQuUbWBA7Se968eeV+vPPOO+aT8o+HHnpIOoR9bLPNNilPOrUwafIdDJ4wsLoOPfRQc+R5Z555prwfwegHTYv24GTHxEYwRiqrYqcLJoSGDRuao9ggnPmt7777rmkpC/0vaJ3aCcCv5aJMjBo1KqGQC4Ngk3bt2pkjT6wnhIi/SCvPN9lkk9CAGHa1xIqLBcKS801kffEbeF+iMOoo5LrAYSzYZNdevXrJuRJA4mfffff16tWrZ45KGDduXMry3LCWbYLqqlWr5DxIWvVjlUCEpB+u7XbbbSe7oOYaBbMfztKlS82z8hOsf1QRunfvrnSHMkdKdhhlTYESF6lEazlSlgWfsda2TWsJWlBIKZ+jjz5a1gdYTKbA4ZNPPmneUQQL7dRgC67v8BvYeE5ryZGKD+KHZvG+PFSpUqXMnkNR4bdpwa60QC1T6y4Ir1NmBD89BRjD0AJH1apVS66XhWvGRl/st0JNvorCtdZWhaz/rV69Ws5FCxZZz7GwttKmTRs1duxY1bt3b9NaxI033ih99vjjjzctpeE+UO6G4rLxSi/RPyjVwgZmlH9KBNeGfhULFs45b9bSwiDIZv/99zdHsWGd6ssvvzRH5ceOa/qmFq5yjdmkkSK8W221lfQDC+uYO+ywg5QIojSWn2eeeUbW82zprIrAuGXtjKKkI0aMUAMHDpTyPKeddpp5h5JzYBNJxqUWMKa1CMawLRSaSxSMwGFjpqgLdWFoqySlAofNmdhxkCJ/FgoMas3JHKUGrb3KhMMCZqLorMmTJ0vBQRa7O3bsaFqVVKJmsmZSoj5VRWDgcC2DBAWtVobMs9KUN7KPwUcNudGjR5eqcxfGtGnT1DnnnCMLtsFqv/G46qqr5PMXLlwoEW6pZPjw4WrQoEFl7g2TG4KFGmLB/kmNMSarWFspI8yIXoy3cRmgrNF3EHRMxonQFp70mVgQxIKCFUu5QnFhck0EAufggw82R+UDZQnlLwiChx1ECfQZMmSIaVUSrYfSoq2LMtXkJ02aJAKcYIxUss8++4hgRdhVrVrVtCq1yy67qK233lqUvSAoTJxjzuGCBrIHvnlyPfRtSNs+Hbgf+XxcmInQgkncM8FFZOvrz+cabHb9g6CIROjJTt6LKyMZmjdvLgEXqVhcD6InHXGnBReh8evXqFHDHJUmUaIvriJ+Z9AlE8S6ZaNcu0TkQ9AAUAKL8wxWWbAbuQX3uQFcmMGAjoqCW5fvw9Xth4AA2sOqZPA/Yfvt5AJZC4s+5ZRT1OGHH17uB9s35ztogWh5uF+ozpsOrKmNNhgPPRFIOCWVbDfbbDPTWoQWWvKX627B0qDScq9evSSkOtfB1QFRQp2jXjM/3Ev2JuIaBTV3v0umPLBVAO4vXFFa6JjWolB2rA+shSAffvihqlOnjjkKx4ZYJ8o1ivq+ygRjAXdpw4YNTUsReArIXwpavoQncz9SvT005wG4Tf1wHtC6dWv56wfrEOssJ3FBA9mBBT20YRaFWbhPFyzm69ucMMQXjY33XXHFFaalBBZJ9eAzR54EDlx11VXyfK+99pJ6cblOo0aNSv2GeFB0lWvhD3sNQhSaP4DCWoGESPshU5+opopA7TE+m63R/dCHaA8LfSaCjsipeNjgiERVBFi85n1aAJuW8pMPFo49R8LRgxByHBagwQJ+vEjQ8kIoNOfyySefmJYiBg8eLO3BIB7OPbh9dS6RNQvngAMOKGWxJPuIsqiYq6A5H3PMMapu3bqijfgrEKSaatWqyT4uwcTAIPiC0cz9PmJYtmyZPFiEtLAoju8brZ51JxZbcxkWgLEQmjZtalrig2XAvkixrhnJnawdoP3aRE/rL8ff7oekvWAAgZ5sZRGe74kSzEI/gWA/secXXHznnNjwL9H+PVh97PvEOk48eB/nSh8pBBgHXJOgtUIQB5ZMWLDDzJkz1QknnGCOUoe998FzsWu/wXPRik+kwI5sUVCVBnIBSqvQMX///XeJRAm6r9IBAnrx4sXmKByiYRDi/p1JWaS0UTF8hgV3KBMVAQREjp1++unmldyESR2hGHWBmYkd91Wsa0ZmNxvZETGHQH/jjTckSIDJwR81deutt8o24EHXFsEIRH6Rxc525okgUgoXMv1FK4nSxvcT5IFL5e2335Y2y5VXXqkGDBhgjmKDCzDMHRfEugoLCYJGCG5Yv369HPOXYBP6OtfD70YmQrBz584yFlINUY8E7BD8Y5kzZ47MGyiH/kAootVQfHJ5I7eCiVLLFYgwIryUCSdKTa9UQLgukTXsghgrFBWwZHhf27ZtRQAxyJhACZtFmw5qzITrotH7w7tzEUJHCSElbJXfFgUit6699lr12WefmZbS3HnnnTL5E66M0CGaCUHCX8JmUSyw4rt06WL+ozRERqExsxbA+SUCq2XkyJHyd5NNNlHff/+9uuGGG0TzZbdXjrG4ED6cd6xQaAvnznkSVcfkGgusQ6LcUC5QNCoK34tAx8rjd+QqrGlSHoboP9au6P9EIRKFxl+EDlYqfxFCffr0Mf+ZelCYUF6oacd6GmuM3HtCy4mgwxrjmiJwGIv+UP2co5Cj1MicJivXVi+gHAmlSNIFvnbdMeLuypkOWCPSE1u5vlcLmdDsaXzHuvsUJwMmG9GVSS666CLZCyiZ5Esyvdk8T0/gpiU9hJUryQTsYEpEohZgpiUcrWxIaaRUrblkYg2HtS02zbPjmh1dtRVqXnVkk4IVOIS+UteMgcfCOqX7tdYpg8EuiKcSrSHKZwcXlTPFmDFjSmWt+2HRmO2EOUc/tmYUlRGCsBNk/fr15TlCJ5cCBxCwBJbYEFXClSkTkizshUOoa7pgET5bO2oed9xxUj0iEewJlOr6eXxmuiAcmFI/3H/GNWV7KGlEP6bmnCO7FKTAWbFihRTDCyusR2elc6ayZhgTMtrkueeea1oyDxoluRzBMh0wcOBA+c1ENlm06e41aNBAcnPCOOOMMyTPg8md3xV2LbMF9cf4PezuyX42PC+PpUI+EiVMUlFDLAwKqhKBlmmInCPaKliUMgh5Ylg3P/zwg2nJbShwipcCK8oPigeRlvSDOXPmmFZHNihIgXPOOedIiCkaUJDZs2dLx8SdUp76V0Ew76mVhoCzNZrSBQMunsbMpIslE4SJecCAAcXnRyVbEgpPP/30UnW7/Lz55psidEaMGJFzVWlxo5BQS+g8VXz9gjRZ2JY8WMMqFaxcuVKub6ZBQcCyv+GGG0xLOPT9Jk2aeGPHjjUtuc8RRxwhhWjDxhlbyzOucRE7skdBChyr7QSzd8FqxDxee+0101o+KFO/xx57iGWxLgP74pNrwkQWD/IFKDLoB43wzjvvlHwL3Cy4IBBe+QqChvL8rJkFc1fKA5Nuomz8ZMG6+f77781R5mCbdfI0glZAEHKH0iFo0wl5bYzbsG3BbQVwHqtXrzatjkxTkAKHtQw6HhNrEMxv2zHD1i6iggvrkEMOkbIj6XLJWHD9HHXUUZF945MnT05J5d9CAiswWOYk38Bao88nWrAn4RXry66B5QtY5YzbsD2mbMVlHm+99ZZpdWSaghQ4bAWAiR227rBs2bLijrl48WLTmjxoh2zelO6KCLjs7J4rydRj8+/l4YhGvl8ztpzApZYIgkgSWUC5CNY9+w6F/Ua2YLDj2vX97FGQAiceuBzolFgn5WXIkCHyGewamC4ov8K6ix1EPDLhtnM48hGiDRkj2Vg3c5TgEj99kOhIciMJkiRmkriXLDNmzJCsY0rGkyCWCshsJ0mQJESSwCidHswuJ3lv6tSp5sjhcFi+/fZb2UuIpGeKmtauXdu84sg0TuD4YG+RMWPGyH4t5alGTYkTSqKQZZ5pyDpmvxeHw1Gabt26yZimNFCq96pxJIerpWZgwqZ8BMU0yyNs2MyKciLZEDZobE7YOBxlYcM0hA21yJywyT7OwtFQSRhhQc0w6iOVByonL1q0yBxlFmpiUSTS4XCU8Prrr0ttQGoXBqt4O7JDwQscqu4ibKZNm5bSLaQdDkf2YK2GLaDZFtut2eQOBe1S+/nnn2WffszuoLChOu5zzz1njhwOR75A1e7zzz9fzZ49u4ywIZgHYeTIDgUrcP7880/Z1nn8+PGh7igCAJybyuHILyjfj7BBidxll11MawnsH8MW0Y7sUJAuNc/zVKdOnWTPizA3GrtYzps3TzQlh8ORH6BEsjPtnnvuWWbTO2Bvm08//VR2MHVkh4IUOOzcd9NNN5mjcMjBYXMlh8ORH6BEzpo1yxyFQwQq2zA7skNBChzcaOyOFw+iWujADocjPxg2bJjsUBoPthk/9thjzZEj07iw6DTCOhCLlAQgsA1xKsHl99JLL6nBgwebFofD4chtXOJnGvnoo4/U/PnzpWROKiBqjgTPoUOHSimbhQsXmlccDocj93EWTp7SokULte2220pdNYfD4cgHnIXjcDgcjozgBE4aWL58ubi8evToof7++2/T6nA4HIWNc6mlGNZtrrzySikC2rhxYzVgwADVvXt3eY1aa3PmzJHnUSE8u127duaoBOdSczgc+YazcFLM6NGj1dixYyU8c/Xq1WqzzTYzr2jpvsEG5ll0yvM/DofDkYs4CyfFPPPMMxLnP3HiREkwXbNmjdpiiy3Mq6nDWTgOhyPfcBZOirFJZey+2aVLl7QIG4fD4chHnIWTBljHofAn+3FgiaxatUq2uHVrOA6Ho5BxFk4amD59uhQQRCiQ+ElVAKBoqMPhcBQqzsJJA+yxgytt0qRJqmfPnrKes+mmm5pXKw514Jo2bSoWzmuvvaaqV69uXnE4HI7cxQmcNMCeG7feeqtq0qSJ6tChQ8r21aGs+owZM1TVqlVNSxGUzunVq5eqV6+eaXE4HI5cQ6n/B94COVCf7VAsAAAAAElFTkSuQmCC)

Максимум последующего задается минимумом: ${\frac{ \beta}{2}\sum_{i=1}^{N}(y_i-f(xᵢ,w))2 + \frac{\alpha}{2} w^ T w}$

## Установка

Для запуска этого кода вам необходимо установить библиотеку Pyro. Вы можете установить ее с помощью следующей команды:

```bash
pip install pyro-ppl
```

## Обзор кода

### 1. Генерация данных

Генерируются симулированные данные с линейной зависимостью, включая случайный шум.

```python
true_slope = 2
true_intercept = 1

X = torch.linspace(0, 10, 100)
Y = true_intercept + true_slope * X + torch.randn(100)
```

### 2. Байесовская модель регрессии

Определяется байесовская модель регрессии с априорными распределениями для наклона, пересечения и стандартного отклонения. Вероятность моделируется с использованием нормального распределения.

```python
def model(X, Y):
    slope = pyro.sample("slope", dist.Normal(0, 10))
    intercept = pyro.sample("intercept", dist.Normal(0, 10))
    sigma = pyro.sample("sigma", dist.HalfNormal(1))

    mu = intercept + slope * X

    with pyro.plate("data", len(X)):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=Y)
```

### 3. Байесовский вывод с использованием SVI

Для байесовского вывода используется стохастическая вариационная инференция (SVI). Определяется функция-руководство для аппроксимации апостериорных распределений параметров модели.

```python
def guide(X, Y):
    slope_loc = pyro.param("slope_loc", torch.tensor(0.0))
    slope_scale = pyro.param("slope_scale", torch.tensor(1.0),
                             constraint=dist.constraints.positive)
    # ... (аналогичные параметры для пересечения и стандартного отклонения)

    slope = pyro.sample("slope", dist.Normal(slope_loc, slope_scale))
    # ... (аналогичная выборка для пересечения и стандартного отклонения)
```

### 4. Обучение модели

Выполняется оптимизация SVI для обучения байесовской регрессионной модели.

```python
optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

num_iterations = 1000

for i in range(num_iterations):
    loss = svi.step(X, Y)
    if (i + 1) % 100 == 0:
        print(f"Iteration {i + 1}/{num_iterations} - Loss: {loss}")
```

### 5. Постериорные образцы

Получаются постериорные образцы с использованием модуля `Predictive`.

```python
predictive = Predictive(model, guide=guide, num_samples=1000)
posterior = predictive(X, Y)

slope_samples = posterior["slope"]
intercept_samples = posterior["intercept"]
sigma_samples = posterior["sigma"]
```

### 6. Оценка параметров

Вычисляются средние значения постериорных образцов для оценки параметров.

```python
slope_mean = slope_samples.mean()
intercept_mean = intercept_samples.mean()
sigma_mean = sigma_samples.mean()

print("Estimated Slope:", slope_mean.item())
print("Estimated Intercept:", intercept_mean.item())
print("Estimated Sigma:", sigma_mean.item())
```

### 7. Визуализация апостериорных распределений

Визуализируются апостериорные распределения наклона, пересечения и стандартного отклонения с использованием графиков плотности ядра.

```python
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

sns.kdeplot(slope_samples, shade=True, ax=axs[0])
axs[0].set_title("Posterior Distribution of Slope")
axs[0].set_xlabel("Slope")
axs[0].set_ylabel("Density")

# ... (аналогичные графики для пересечения и стандартного отклонения)

plt.tight_layout()
plt.show()
```

![example image](https://github.com/denis-samatov/Bayesian_regression_model/blob/main/img.png)

## Результаты

### Преимущества байесовской регрессии:

- **Эффективность при небольших объемах данных:** Байесовская регрессия оказывается очень эффективной, когда у нас мало данных.
- **Подходит для онлайн-обучения:** Особенно хорошо подходит для онлайн-обучения, когда данные поступают в режиме реального времени, по сравнению с пакетным обучением.
- **Надежность математически:** Байесовский подход является проверенным и математически надежным методом, который можно использовать даже без предварительных знаний о данных.
- **Возможность использования внешней информации:** Методы байесовской регрессии используют искаженные расп

ределения, что позволяет включать в модель внешнюю информацию.

### Недостатки байесовской регрессии:

- **Время вывода модели:** Вывод модели может занять много времени.
- **Неэффективность при больших объемах данных:** Если у нас есть большой объем данных, байесовский подход может быть неэффективным по сравнению с частотным подходом.
- **Сложности с установкой пакетов:** Если установка новых пакетов представляет трудности, это может стать проблемой.
- **Зависимость от линейности:** Байесовские модели также подвержены ошибкам, присущим традиционным моделям частотной регрессии, и они все еще зависят от линейности взаимосвязей между характеристиками и переменной результата.

### Когда использовать байесовскую регрессию:

- **Небольшой размер выборки:** Байесовский вывод особенно полезен, когда у нас маленький объем данных. Это хороший выбор, если нужно разработать сложную модель, но данных ограничено.
- **Надежные предварительные знания:** Простой способ внедрить надежные внешние знания в модель - использовать байесовскую модель. Влияние предварительных знаний будет более заметным при работе с небольшими наборами данных.
