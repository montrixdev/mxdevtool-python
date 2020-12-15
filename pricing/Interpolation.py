# excel link : https://blog.naver.com/montrix/221410043168

import mxdevtool as mx

def test():
    print('interpolation test...')

    # 1 dimension
    print('1 dim ---------')
    data = [(1.0, 6.0), 
            (2.0, 5.0), 
            (3.0, 8.0), 
            (4.0, 6.0), 
            (5.0, 4.0), 
            (6.0, 1.0), 
            (7.0, 2.0), 
            (8.0, 3.0), 
            (9.0, 6.0), 
            (10.0, 5.0), 
            (11.0, 4.0), 
            (12.0, 2.0)]  

    x = [v[0] for v in data]
    y = [v[1] for v in data]

    interpolation1d = mx.Interpolation1D(mx.Interpolator1D.ForwardFlat, x, y)

    print(interpolation1d.interpolate(1.17))
    print(interpolation1d.interpolate([1.17, 2,23]))

    # 2 dimension
    print('2 dim ---------')
    z = mx.Matrix(len(x), len(y))

    for i in range(len(x)):
        for j in range(len(y)):
            z[i][j] = i*(pow(j, 0.5))

    interpolation2d = mx.Interpolation2D(mx.Interpolator2D.Bilinear, x, y, z)

    print(interpolation2d.interpolate(1.17, 2.33))
    print(interpolation2d.interpolate([1.17, 2.23], [4.3, 3.3]))

    # 이거 interpolation 2개짜리 엑셀 파일 확인해야함.
    
if __name__ == "__main__":
    test()