import sys
import json
import pandas as pd
import numpy as np
import simplejson
from scipy.stats import norm
from scipy import stats
from scipy.stats.mstats import gmean

def set_data(mystr):
    
    df = pd.read_json(in_str)
    df.set_index(['date'], inplace=True)

    # 데이터 값을 float로 변환
    df['backtest'] = df['backtest'].astype(float)
    df['benchmark'] = df['benchmark'].astype(float)
    df['kospi'] = df['kospi'].astype(float)
    df['riskfree'] = df['riskfree'].astype(float) / 100

    df = df.sort_index()
    df.index = pd.to_datetime(df.index)

    # 월단위 데이터로 resampleing
    df = df.resample(rule='M').last()

    # 직전일의 가격을 'prev'에 저장. 지수값은 직전일부터 필요하고, 변동률은 당일부터 필요한데, 중복관리시 계산과정이 복잡해짐.
    # 날짜 count를 맞추기 위한 작업임
    df['prev'] = df['backtest'].shift(1)
    df['rtn'] = df['backtest'].pct_change()
    df['bm_prev'] = df['benchmark'].shift(1)
    df['bm_rtn'] = df['benchmark'].pct_change()
    df['kospi_rtn'] = df['kospi'].pct_change()
    
    df = df.iloc[1:]
    #df = df['2001-01':'2019-04']
    
    return df

def init_result():
    return {
    'final_balance' : 0,
    'cagr' : 0,
    'stdev' : 0,
    'annlzd_stdev' : 0,
    'arith_mean' : 0,
    'annlzd_arith_mean' : 0,
    'geo_mean' : 0,
    'annlzd_geo_mean' : 0,
    'vol' : 0,
    'annlzd_vol' : 0,
    'hist_var' : 0,
    'anal_var' : 0,
    'c_var' : 0,
    'best_y' : {'year' : 0, 'rtn' : 0},
    'worst_y' : {'year' : 0, 'rtn' : 0},
    'mdd' : 0,
    'skewness' : 0,
    'kurtosis' : 0,
    'sharpe_rto' : 0,
    'sortino_rto' : 0,
    'down_dev' : 0,
    'vs_market' : {'beta' : 0, 'alpha' : 0, 'r2' : 0, 'corr' : 0},
    'vs_benchmark' : {'beta' : 0, 'alpha' : 0, 'r2' : 0, 'corr' : 0}
    }

def analyze_data(df):
    
    result = init_result()
    
    # 1. Final Balance
    initial_balance = 1000000
    final_balance = initial_balance * df['backtest'].iloc[-1] / df['prev'].iloc[0]
    result['final_balance'] = final_balance

    # 2. CAGR (2000-12월 종가부터, 2019-04월 종가까지 기간을 잡는다.)
    year = df['backtest'].count() / 12
    # 2000-12월 종가는 2001-1월의 전일가격을 사용
    CAGR = ( df.iloc[-1]['backtest'] / df.iloc[0]['prev'] )**(1/year) - 1
    result['cagr'] = CAGR

    # 3. Stdev (Annualized standard deviation of monthly returns)
    stdev = df['rtn'].std()
    result['stdev'] = stdev
    # 연간화를 위해 루트12 를 곱한다.
    annlzd_stdev = stdev*(12**0.5)
    result['annlzd_stdev'] = annlzd_stdev
    result['vol'] = stdev
    result['annlzd_vol'] = annlzd_stdev

    # 4. Arithmetic Mean (monthly). 
    arith_mean = df['rtn'].mean()
    result['arith_mean'] = arith_mean

    # 5. Arithmetic Mean (annualized).
    annualized_arith_mean = (1 + arith_mean) ** 12 - 1
    result['annlzd_arith_mean'] = annualized_arith_mean

    # 6. Geometric Mean, scipy 의 gmean사용
    # 수익률의 기하평균은 각 수익률에 1을 더한후 루트를 적용, 이후에 1을 뺀다
    # monthly_rtn의 모든 컬럼값에 1을 더한다
    df['rtn_1'] = df['rtn'] + 1
    # gmean은 list형의 인자를 받는다
    geo_mean = gmean(df['rtn_1'].tolist()) - 1
    result['geo_mean'] = geo_mean

    # 7. Geometric Mean(annualized)
    annualized_geo_mean = ( 1 + geo_mean) ** 12 - 1
    result['annlzd_geo_mean'] = annualized_geo_mean

    # 8. Volatility (monthly) . 변동성은 표준편차를 의미
    #stdev = m_idx['rtn'].std() 
    #result['stdev'] = stdev

    # 9. Volatility (annualized). 3에서 구한 Stdev와 같은 값이다
    # 연간화를 위해 루트12 를 곱한다.
    #stdev = stdev*(12**0.5)

    # 10. VaR
    # 10.1 Historical VaR 
    # exclusive quantile을 자체 구현
    def quantile_exc(df2, q):
        list_sorted = sorted(df2) # sorted()는 list형의 결과를 리턴한다
        rank = q * (len(list_sorted) + 1) - 1
        #print ("q_exc : ", rank)
        #assert rank > 0, 'quantile is too small'
        if rank < 0 :
            print ('quantile is too small')
            return 0
        rank_l = int(rank)
        return list_sorted[rank_l] + (list_sorted[rank_l + 1] - 
                                      list_sorted[rank_l]) * (rank - rank_l)

    historical_var_95 = quantile_exc(df['rtn'], 0.05)
    if (historical_var_95 == 0) :
        historical_var_95 = df['rtn'].quantile(0.05)
    result['hist_var'] = historical_var_95

    # 10.2 Analytical VaR
    mean = df['rtn'].mean()
    stdev = df['rtn'].std()
    analytical_var_95 = norm.ppf(0.05, mean, stdev)
    result['anal_var'] = analytical_var_95

    # 10.3 Conditional VaR
    # 자체구현
    def conditional_var(df3, q):
        list_sorted = sorted(df3)
        rank = q * len(list_sorted) 
        rank_l = int(rank)

        sum_rtn = 0
        sum_rtn = sum(i for i in list_sorted[0:rank_l])

        return 1 / rank * sum_rtn

    cvar_95 = conditional_var(df['rtn'], 0.05)
    result['c_var'] = cvar_95

    # 11. Best Year / Worst Year
    # 년단위 데이터로 resamplingn
    y_idx = df.resample(rule='Y').last()
    y_idx['rtn'] = y_idx['backtest'].pct_change()

    # 1년이내의 데이터의 경우 y_idx['rtn'] 값이 null 이므로 아래와 같이 계산
    if len(y_idx) == 1 :
        min_val = df['backtest'].iloc[-1] / df['prev'].iloc[0] - 1
        min_idx = df['backtest'].idxmin()
        max_val = df['backtest'].iloc[-1] / df['prev'].iloc[0] - 1
        max_idx = df['backtest'].idxmax()        
    else:     
        min_val = y_idx['rtn'].min()
        min_idx = y_idx['rtn'].idxmin()
        max_val = y_idx['rtn'].max()
        max_idx = y_idx['rtn'].idxmax()
        
    result['best_y']['year'] = max_idx.year
    result['best_y']['rtn'] = max_val
    result['worst_y']['year'] = min_idx.year
    result['worst_y']['rtn'] = min_val

    # 12. MDD, 
    # - step1.지수의 수익률을 일별 누적(1+r을 계속곱해나감). 
    # - step2. 누적수익률에 대한 MAX를 일별로 기록
    # - step3. 일별로 누적수익률과 MAX수익률 간의 차이((CUM - MAX) / MAX) 가 가장 큰 것을 잡는다.

    #  등락률에 1을 더한다
    df['rtn_1'] = df['rtn'] + 1

    # 누적수익률계산
    df['cum'] = df['rtn_1'].cumprod()

    # 누적수익률중 최고값
    df['high'] = df['cum'].cummax()

    # drawdown 계산
    df['drawdown'] = (df['cum'] - df['high'])/df['high']
    MDD = df['drawdown'].min()
    result['mdd'] = MDD

    # 13. Skewness
    skewness = df['rtn'].skew()
    result['skewness'] = skewness

    # 14. Excess Kurtosis
    ex_kurtosis = df['rtn'].kurtosis()
    result['kurtosis'] = ex_kurtosis

    # 15. Ratio
    # https://www.quantnews.com/performance-metrics-sharpe-ratio-sortino-ratio/
    # 15.1 Sharpe Ratio
    # denominator - month(12), day(252)
    denominator = 12
    df['excess_rtn'] = df['rtn'] - df['riskfree']/denominator
    sharpe_rto = df['excess_rtn'].mean() /  df['excess_rtn'].std() * np.sqrt(denominator)
    result['sharpe_rto'] = sharpe_rto

    # 15.2 Sortino Ratio
    target = 0
    df['downside_rtn'] = 0
    df.loc[df['rtn'] < target, 'downside_rtn'] = df['rtn']**2
    down_stdev = np.sqrt(df['downside_rtn'].mean())
    sortino_ratio = df['excess_rtn'].mean()/down_stdev * np.sqrt(denominator)
    result['sortino_rto'] = sortino_ratio
    result['down_dev'] = down_stdev

    # downside_stdev 를 excess_rtn으로 계산
    #m_idx['downside_rtn2'] = 0
    #m_idx.loc[m_idx['excess_rtn'] < target, 'downside_rtn2'] = m_idx['excess_rtn']**2
    #down_stdev2 = np.sqrt(m_idx['downside_rtn2'].mean())
    #sortino_ratio = m_idx['excess_rtn'].mean()/down_stdev2 * np.sqrt(denominator)

    # 16. [vsMarket] Beta, Alpha, R-squared, correlation
    # Beta, Alpha, R squared 참고사이트
    # http://gouthamanbalaraman.com/blog/calculating-stock-beta.html
    # https://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy

    # 16.1 Beta
    covariance = np.cov(df['rtn'], df['kospi_rtn'])
    # variance는 np.var로 구할수도 있으나, covariance[1,1] 과 같다
    #variance = np.var(m_idx['mkt_rtn'],ddof=1)
    beta = covariance[0,1] / covariance[1,1]
    result['vs_market']['beta'] = beta

    # 16.2 Alpha
    alpha = df['rtn'].mean() - beta*(df['kospi_rtn'].mean())
    #연환산
    y_alpha = (1 + alpha) ** 12 - 1
    result['vs_market']['alpha'] = y_alpha

    # 16.3 R squared 
    # R2 - numpy_manual

    ypred = alpha + beta * df['kospi_rtn']
    SS_res = np.sum(np.power(ypred - df['rtn'],2))
    SS_tot = covariance[0,0] * (len(df) - 1) # SS_TOT is sample_variance*(n-1)
    r_squared = 1. - SS_res/SS_tot
    result['vs_market']['r2'] = r_squared

    # 1year momentum (bonus) 
    momentum = np.prod(1+df['rtn'].tail(12).values) - 1

    # 16.4 correlation
    # 비교를 위해 'rtn', 'mkt_rtn'만 새로운 dataframe 으로 copy
    #new_df = m_idx[['rtn','mkt_rtn']].copy()
    #corr = new_df.corr()
    corr = df['rtn'].corr(df['kospi_rtn'])
    result['vs_market']['corr'] = corr
    
    if 'benchmark' in df.columns:
        
        # 17. [vsBenchmark] Beta, Alpha, R-squared, correlation
        # Beta, Alpha, R squared 참고사이트
        # http://gouthamanbalaraman.com/blog/calculating-stock-beta.html
        # https://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy

        # 17.1 Beta
        covariance = np.cov(df['rtn'], df['bm_rtn'])
        # variance는 np.var로 구할수도 있으나, covariance[1,1] 과 같다
        #variance = np.var(m_idx['mkt_rtn'],ddof=1)
        beta = covariance[0,1] / covariance[1,1]
        result['vs_benchmark']['beta'] = beta

        # 17.2 Alpha
        alpha = df['rtn'].mean() - beta*(df['bm_rtn'].mean())
        #연환산
        y_alpha = (1 + alpha) ** 12 - 1
        result['vs_benchmark']['alpha'] = y_alpha

        # 17.3 R squared 
        # R2 - numpy_manual

        ypred = alpha + beta * df['bm_rtn']
        SS_res = np.sum(np.power(ypred - df['rtn'],2))
        SS_tot = covariance[0,0] * (len(df) - 1) # SS_TOT is sample_variance*(n-1)
        r_squared = 1. - SS_res/SS_tot
        result['vs_benchmark']['r2'] = r_squared

        # 17.4 correlation
        # 비교를 위해 'rtn', 'mkt_rtn'만 새로운 dataframe 으로 copy
        #new_df = m_idx[['rtn','mkt_rtn']].copy()
        #corr = new_df.corr()
        corr = df['rtn'].corr(df['bm_rtn'])
        result['vs_benchmark']['corr'] = corr
    
    return result
    
def show_result(rslt) :
    for key in rslt.keys() :
        print("Final Balance : " , int(rslt[key]['final_balance']))
        print("CAGR : ", round(rslt[key]['cagr'] * 100, 5), "%" )
        print("Stdev : ", round(rslt[key]['annlzd_stdev'] * 100, 5), "%" )
        print("Best Year (",rslt[key]['best_y']['year'],") : ", round(rslt[key]['best_y']['rtn'] * 100, 5), "%" )
        print("Worst Year (",rslt[key]['worst_y']['year'],") : ", round(rslt[key]['worst_y']['rtn'] * 100, 5), "%" )
        print("MDD : ", round(rslt[key]['mdd'] * 100, 5), "%" )
        print("Sharpe Ratio : ", round(rslt[key]['sharpe_rto'], 5))
        print("Sortino Ratio : ", round(rslt[key]['sortino_rto'], 5))
        print("Korean MKT Correlation : ", round(rslt[key]['vs_market']['corr'], 5))
        print("Arithmetic Mean (daily) : ", round(rslt[key]['arith_mean'] * 100, 5), "%" )
        print("Arithmetic Mean (annualized) : ", round(rslt[key]['annlzd_arith_mean'] * 100, 5), "%" )
        print("Geometric Mean (daily) : ", round(rslt[key]['geo_mean'] * 100, 5), "%" )
        print("Geometric Mean (annualized) : ", round(rslt[key]['annlzd_geo_mean'] * 100, 5), "%" )
        print("Volatility (daily) : ", round(rslt[key]['stdev'] * 100, 5), "%" )
        print("Volatility (annualized) : ", round(rslt[key]['annlzd_stdev'] * 100, 5), "%" )
        print("Downside Deviation (daily) : ", round(rslt[key]['down_dev'] * 100, 5), "%" )
        print("MDD : ", round(rslt[key]['mdd'] * 100, 5), "%" )
        print("Korean MKT Correlation : ", round(rslt[key]['vs_market']['corr'], 5))
        print("Beta(vs market) : ", round(rslt[key]['vs_market']['beta'], 5))
        print("Alpha(vs market, annualized) : ", round(rslt[key]['vs_market']['alpha']*100, 5),"%")
        print("R2(vs market) : ", round(rslt[key]['vs_market']['r2']*100, 5),"%")
        print("Beta(vs benchmark) : ", round(rslt[key]['vs_benchmark']['beta'], 5))
        print("Alpha(vs benchmark, annualized) : ", round(rslt[key]['vs_benchmark']['alpha']*100, 5),"%")
        print("R2(vs benchmark) : ", round(rslt[key]['vs_benchmark']['r2']*100, 5),"%")
        print("Sharpe Ratio : ", round(rslt[key]['sharpe_rto'], 5))
        print("Sortino Ratio : ", round(rslt[key]['sortino_rto'], 5))
        print("Skewness : ", round(rslt[key]['skewness'], 5))
        print("Excess Kurtosis : ", round(rslt[key]['kurtosis'], 5))
        print("Historical VaR(5%) : ", round(rslt[key]['hist_var']*100, 5),"%")
        print("Analytical VaR(5%) : ", round(rslt[key]['anal_var']*100, 5),"%")
        print("Conditional VaR(5%) : ", round(rslt[key]['c_var']*100, 5),"%")
        print("="*50)
    
if __name__ == '__main__':
    
    rslt = dict.fromkeys(['backtest','benchmark'])
    
    #json을 읽어서, 스트링으로 변환한다. 나중에 해당 스트링을 argv로 받을 예정
    filename = sys.argv[1]
    in_str = open(filename).read()
    
    #배포시 아래로 변경
    #in_str = sys.argv[1]
    
    tm_data = set_data(in_str)
    
    rslt['backtest'] = analyze_data(tm_data)
    
    if 'benchmark' in tm_data.columns:
        tm_data['backtest'] = tm_data['benchmark']
        tm_data['rtn'] = tm_data['bm_rtn']
        tm_data['prev'] = tm_data['bm_prev']
        rslt['benchmark'] = analyze_data(tm_data)
    #show_result(rslt)    
    rslt_json = simplejson.dumps(rslt, ignore_nan=True)
    print(rslt_json)