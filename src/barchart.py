from axis import *
import math
import random

random.seed(4) # 42)

DEBUG = True

def scrape_vertical_bar_chart(imgfile,
    angles_x=[0,-45,-90], round_y = 4, show=False):
    '''
    End-to-end function for parsing a regular
    vertical bar chart from an image
    '''
    img = cv2.imread(imgfile)
    assert show
    assert os.path.exists(imgfile)
    result = ocr_2_axes(imgfile, angles_x=angles_x)
    axis_x = result['axis_x']
    axis_y = result['axis_y']
    # print('x-axis:')
    # print(axis_x)
    # print(axis_x['text'].tolist())
    # print('y-axis:')
    # print(axis_y)
    # print(axis_y['text'].tolist())
    lines = find_lines(imgfile)
    x_min = axis_x['tlx'].min()
    x_max = axis_x['brx'].max()
    y_min = axis_y['top'].min()
    y_max = axis_y['bottom'].max()
    dx_per_categ = round((x_max - x_min) / len(axis_x))
    # print("Pixel per category (x):", dx_per_categ)
    # print("xmin,max", x_min, x_max)
    hlines = lines[(lines['x1'] >= x_min) & (lines['x2'] >= x_min) & \
        (lines['x1'] <= x_max) & (lines['x2'] <= x_max) & \
        (lines['y1'] >= y_min) & (lines['y2'] >= y_min) & \
        (lines['y1'] <= y_max) & (lines['y2'] <= y_max) & \
        (np.abs(lines['angle']) < 5.0) & \
        (lines['Length'] < 0.5*dx_per_categ)].reset_index(drop=True)
    if show:
        img2 = copy.deepcopy(img)
        for i in range(len(hlines)):
            start = (hlines.at[i,'x1'], hlines.at[i,'y1'])
            end = (hlines.at[i,'x2'], hlines.at[i,'y2'])
            cv2.line(img2,start, end, (0,0,255),2)
        cv2.imshow('horizonal lines', img2)
        if WAITKEY:
            cv2.waitKey(0)
    # print(result)
    xlabels = []
    xcharts = []
    ycharts = []
    xms = []
    yms = []
    for i in range(len(hlines)):
        xm = 0.5*(hlines.at[i,'x1'] + hlines.at[i,'x2'])
        ym = 0.5*(hlines.at[i,'y1'] + hlines.at[i,'y2'])
        xchart = predict_point(result['px2chart_x'], xm)
        ychart = round(predict_point(result['px2chart_y'], ym), round_y)
        if math.isnan(xchart):
            print("warning: x-position was NaN x:", xchart, 'y:', ychart)
            continue
        xlevel = round(xchart)
        # make sure not out of index bounds:
        if xlevel < 0:
            xlevel = 0
        if 'levels_x' in result:
            if xlevel >= len(result['levels_x']):
                xlevel = len(result['levels_x']) - 1
            xlabel = result['levels_x'][xlevel]
        else:
            xlabel = xlevel
            print("warning: looks like special case where x-axis appears numeric but should be interpreted as factor variable. Currently treating as numeric TODO")
        # print("prediction:", xchart, xlabel, ychart)
        xcharts.append(xchart)
        ycharts.append(ychart)
        xlabels.append(xlabel)
        xms.append(xm)
        yms.append(ym)
    df_chart = pd.DataFrame(data={'x':xlabels,'y':ycharts,
        'xpos':xms, 'ypos':yms}).sort_values('xpos').reset_index(drop=True)
    # print(df_chart)
    return {'chart_df':df_chart, 'axes':result }


def test_scrape_vertical_bar(imgfile= \
    join(RAW_DIR,'train','images','02f76c3816a0.jpg'), # 53c7b64a0cba.jpg'), # 3bb215f4daea.jpg'), # 'aaeeb3e6866d.jpg'),
    annofile= \
    join(RAW_DIR,'train','annotations','02f76c3816a0.json'), # 53c7b64a0cba.json'), # 3bb215f4daea.json'), # 'aaeeb3e6866d.json'),
    angles_x=[0,-45,-90],
    show=True):
    # print("input file:", imgfile)
    assert os.path.exists(imgfile)
    assert os.path.exists(annofile)
    result = scrape_vertical_bar_chart(imgfile, angles_x=angles_x, show=show)
    # print('result keys', list(result.keys()))
    axis_x = result['axes']['axis_x']
    axis_y = result['axes']['axis_y']
    print('x-axis:')
    print(axis_x)
    print(axis_x['text'].tolist())
    print('y-axis:')
    print(axis_y)
    print(axis_y['text'].tolist())
    # Load JSON file
    anno = read_annotation(annofile)
    # with open(annofile, 'r') as file:
    #    anno = json.load(file)
    data_series = anno['data-series']
    print('prediction:')
    print(result['chart_df'])
    chart = result['chart_df']
    print(data_series)
    rms= 0
    missing_count = 0
    residual = 0
    for i in range(len(data_series)):
        label = data_series.at[i,'x']
        y = data_series.at[i,'y']
        found = False
        for j in range(len(chart)):
            label2 = chart.at[j,'x']
            y2 = chart.at[j,'y']
            if label2 == label:
                rms += (y-y2)*(y-y2)
                residual += (y2-y)
                found = True
                break
        if not found:
                missing_count += 1
    print("missing:", missing_count, round(100*missing_count/len(data_series),1),'%')
    if missing_count < len(data_series):
        rms = rms / (len(data_series)-missing_count)
        residual /= (len(data_series)-missing_count)
    rms = round(math.sqrt(rms), 4)
    print('RMS distance:', rms, 'Residual:', round(residual,4))
    assert rms < 5.0 # TODO decrease threshold over time
    assert axis_y['text'].tolist()[0] == '20'
    assert axis_y['text'].tolist()[5] == '120'
    assert len(axis_x) == 10
    assert len(axis_y) == 6
    return {'rms':rms, 'residual_mean':residual, 'n':len(data_series), 'missing':missing_count }


def run_scrape_vertical_bar(imgfile= \
    join(RAW_DIR,'train','images','aaeeb3e6866d.jpg'),
    annofile= \
    join(RAW_DIR,'train','annotations','aaeeb3e6866d.json'),
    angles_x=[0,-45,-90],
    show=True):
    '''
    Runs content extraction for image of bar chart
    and performs error estimation using ground-truth data
    '''
    # print("input file:", imgfile)
    assert os.path.exists(imgfile)
    assert os.path.exists(annofile)
    result = scrape_vertical_bar_chart(imgfile, angles_x=angles_x, show=show)
    # axis_x = result['axes']['axis_x']
    # axis_y = result['axes']['axis_y']
    # Load JSON file
    anno = read_annotation(annofile)
    # with open(annofile, 'r') as file:
    #    anno = json.load(file)
    data_series = anno['data-series']
    chart = result['chart_df']
    rms= 0
    missing_count = 0
    residual = 0
    for i in range(len(data_series)):
        label = data_series.at[i,'x']
        y = data_series.at[i,'y']
        found = False
        for j in range(len(chart)):
            label2 = chart.at[j,'x']
            y2 = chart.at[j,'y']
            if label2 == label:
                rms += (y-y2)*(y-y2)
                residual += (y2-y)
                found = True
                break
        if not found:
                missing_count += 1
    print("missing:", missing_count, round(100*missing_count/len(data_series),1),'%')
    if missing_count < len(data_series):
        rms = rms / (len(data_series)-missing_count)
        residual /= (len(data_series)-missing_count)
    rms = round(math.sqrt(rms), 4)
    # print('RMS distance:', rms, 'Residual:', round(residual,4))
    return {'rms':rms, 'residual_mean':residual, 'n':len(data_series), 'missing':missing_count }


def runall_scrape_vertical_bar(imgdir= \
    join(RAW_DIR,'train','images'), # ,'aaeeb3e6866d.jpg'),
    annodir= \
    join(RAW_DIR,'train','annotations'), # ,'aaeeb3e6866d.json'),
    angles_x=[0,-45,-90], n=-1,
    chart_types=None, # ['vertical_bar'],
    show=True):
    '''
    Extract data from all files in directories that correspond
    to vertical bar charts.

    Args:
        chart_types(list): None or Skip if 'chart-type' is not in this list
    '''
    dir_list = os.listdir(imgdir)
    rmsv = []
    residuals = []
    basev = []
    nv = []
    missingv = []
    failures = []
    nmax = len(dir_list)
    if n > 0: # do not compute all entries
        nmax = min(n, nmax)
        random.shuffle(dir_list)
    for i in range(nmax):
        file = dir_list[i]
        base = file[:-4]
        afile = join(imgdir, file)
        # print(afile)
        assert os.path.exists(afile)
        jfile = join(annodir, base + ".json")
        with open(jfile, 'r') as f:
            anno = json.load(f)
        # print(anno)
        chart_type = anno['chart-type']
        if chart_types is not None and not chart_type in chart_types:
            continue
        try:
            result = run_scrape_vertical_bar(imgfile= afile,
                annofile = jfile,
                angles_x=angles_x,
                show=show)
        except ValueError:
            print("cannot parse case", base, 'skipping')
            failures.append(base)
            continue
        rmsv.append(result['rms'])
        residuals.append(result['residual_mean'])
        missingv.append(result['missing'])
        nv.append(result['n'])
        basev.append(base)
        print('@result', base, 'mean squared error:', result['rms'], 'mean error:', result['residual_mean'])
        # Close the window and remove the image
        cv2.destroyAllWindows()
    df = pd.DataFrame(data={'ID':basev, 'RMS':rmsv,
        'Residual':residuals,
        'N':nv,
        'Missing':missingv})
    print(df.to_string())
    outfile = "tmp_barchart_vertical_results.csv"
    print("writing to output file", outfile)
    df.to_csv(outfile)
    print("failed cases:", failures)
    print("bye")    

def run_all_tests():
    runall_scrape_vertical_bar(n=100)
    test_scrape_vertical_bar()

    assert False

    

if __name__ == '__main__':
    run_all_tests()

