import os
import gc
import cv2
import math
import time
import joblib
import numpy as np
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
GENERATED_FOLDER = os.path.join(BASE_DIR, 'static', 'generated')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER

if not os.path.exists(UPLOAD_FOLDER) :
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(GENERATED_FOLDER) :
    os.makedirs(GENERATED_FOLDER)

scaler1 = joblib.load(os.path.join(BASE_DIR, 'scaler1.pkl'))
model1 = joblib.load(os.path.join(BASE_DIR, 'model1.pkl'))

def show_image(image) :
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def show_line(image, line) :
    s = image.copy()
    cv2.line(s, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
    show_image(s)
    
def swap(var1, var2) :
    return var2, var1

"""
features to classify lines for triangles
1. length_diff
2. angle_between_lines
3. cosine_similarity
4. endpoint_distance
5. slope
"""

def line_length(line) :
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def angle_between_lines(line1, line2) :
    x1, y1, x2, y2 = line1
    fx1, fy1, fx2, fy2 = line2
    
    if x1 > x2 :
        x1, x2 = swap(x1, x2)
        y1, y2 = swap(y1, y2)
    if fx1 > fx2 :
        fx1, fx2 = swap(fx1, fx2)
        fy1, fy2 = swap(fy1, fy2)
    
    vec1 = np.array([x2 - x1, y2 - y1])
    vec2 = np.array([fx2 - fx1, fy2 - fy1])
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cos_theta = dot_product / norm_product
    
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def cosine_similarity(line1, line2) :
    vec1 = np.array([line1[2] - line1[0], line1[3] - line1[1]])
    vec2 = np.array([line2[2] - line2[0], line2[3] - line2[1]])
    
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def endpoint_distance(line1, line2) :
    d1 = np.linalg.norm(np.array(line1[:2]) - np.array(line2[:2]))
    d2 = np.linalg.norm(np.array(line1[:2]) - np.array(line2[:2]))
    d3 = np.linalg.norm(np.array(line1[2:]) - np.array(line2[:2]))
    d4 = np.linalg.norm(np.array(line1[2:]) - np.array(line2[2:]))
    
    return min(d1, d2, d3, d4)

def slope(line) :
    x1, y1, x2, y2 = line
    if(x2 == x1) :
        return 1e6
    
    return (y2 - y1) / (x2 - x1)

def findMainContours(image, threshold) :
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    main_contour_area = 0.0
    main_approx = None
    main_contour = None
    
    for contour in contours :
        area = cv2.contourArea(contour)
        main_approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if area > main_contour_area and len(main_approx) in [3, 4] :
            main_contour = contour
            main_contour_area = area
    
    return main_contour

def is_horizontally_overlapping(line1, line2, y_thresh=6) :
    x1_1, y1_1, x2_1, y2_1 = line1
    x1_2, y1_2, x2_2, y2_2 = line2
    
    y_avg1 = (y1_1 + y2_1) / 2
    y_avg2 = (y1_2 + y2_2) / 2
    
    if abs(y_avg1 - y_avg2) > y_thresh :
        return False
    
    x_min1, x_max1 = sorted([x1_1, x2_1])
    x_min2, x_max2 = sorted([x1_2, x2_2])

    return not (x_max1 < x_min2 or x_max2 < x_min1)

def is_vertically_overlapping(line1, line2, x_thresh=5) :
    x1_1, y1_1, x2_1, y2_1 = line1
    x1_2, y1_2, x2_2, y2_2 = line2
    
    x_avg1 = (x1_1 + x2_1) / 2
    x_avg2 = (x1_2 + x2_2) / 2
    
    if abs(x_avg1 - x_avg2) > x_thresh :
        return False
    
    y_min1, y_max1 = sorted([y1_1, y2_1])
    y_min2, y_max2 = sorted([y1_2, y2_2])
    
    return not (y_max1 < y_min2 or y_max2 < y_min1)

def rotate_image(image, line) :
    x1, y1, x2, y2 = line
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg = math.degrees(angle_rad)
    
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), borderValue=(255, 255, 255))
    
    return rotated

def funcTriangles(image, threshold, approx) :
    vertices = [tuple(point[0]) for point in approx]

    sides = [(vertices[i][0], vertices[i][1], vertices[(i + 1) % len(vertices)][0], vertices[(i + 1) % len(vertices)][1])
            for i in range(len(vertices))]
    sides = [(x1, y1, x2, y2) if y1 < y2 or (y1 == y2 and x1 < x2) else (x2, y2, x1, y1)
            for x1, y1, x2, y2 in sides]

    base = min(sides, key=lambda side: abs(slope(side)))
    base_index = sides.index(base)
    if base[0] > base[2] :
        base = (base[2], base[3], base[0], base[1])
    other_sides = [sides[i] for i in range(len(vertices)) if i != base_index]

    lines = cv2.HoughLinesP(threshold, 2, np.pi / 180, threshold=50, minLineLength=0, maxLineGap=10)

    lines = [
        (x1, y1, x2, y2) if (y1 < y2) or (y1 == y2 and x1 < x2) else (x2, y2, x1, y1)
        for [[x1, y1, x2, y2]] in lines
    ]
    
    lines = sorted(lines, key=lambda line: line_length(line), reverse=True)
    
    horizontal_lines = []
    dividing_lines = []
    
    horizontal_lines.append(base)
    
    for side in other_sides :
        dividing_lines.append(side)
    
    if lines is not None :
        for line1 in lines :
            if line_length(line1) < line_length(base) / 20 :
                continue
            keep = True
            hor = None
            x1, y1, x2, y2 = line1
            
            if cv2.pointPolygonTest(np.array(approx), (float((x1 + x2) / 2), float((y1 + y2) / 2)), False) >= 0 :
                if abs(y1 - y2) <= 10 :
                    hor = True
                    for line2 in horizontal_lines :
                        fx1, fy1, fx2, fy2 = line2
                
                        length_diff = abs(line_length(line1) - line_length(line2))
                        angle_diff = angle_between_lines(line1, line2)
                        cos_sim = cosine_similarity(line1, line2)
                        endpoint_dist = endpoint_distance(line1, line2)
                        slope_diff = abs(slope(line1) - slope(line2))
                
                        input_data = [[fx1, fy1, fx2, fy2, x1, y1, x2, y2, length_diff, angle_diff, cos_sim, endpoint_dist, slope_diff]]
                        input_data = scaler1.transform(input_data)
                        prediction = model1.predict(input_data)

                        if prediction[0] == 0 or is_horizontally_overlapping(line1, line2) :
                            keep = False
                            break
                else :
                    hor = False
                    for line2 in dividing_lines :
                        fx1, fy1, fx2, fy2 = line2
                        
                        length_diff = abs(line_length(line1) - line_length(line2))
                        angle_diff = angle_between_lines(line1, line2)
                        cos_sim = cosine_similarity(line1, line2)
                        endpoint_dist = endpoint_distance(line1, line2)
                        slope_diff = abs(slope(line1) - slope(line2))
                        
                        input_data = [[fx1, fy1, fx2, fy2, x1, y1, x2, y2, length_diff, angle_diff, cos_sim, endpoint_dist, slope_diff]]
                        input_data = scaler1.transform(input_data)
                        prediction = model1.predict(input_data)
                        
                        if prediction[0] == 0 :
                            keep = False
                            break
                        
                if keep :
                    if hor :
                        horizontal_lines.append(line1)
                    else :
                        dividing_lines.append(line1)
    
    # วาดเส้นลงบนภาพที่จะใช้แสดงในกรอบ
    ret_image = image.copy()
    for x1, y1, x2, y2 in horizontal_lines :
        cv2.line(ret_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for x1, y1, x2, y2 in dividing_lines :
        cv2.line(ret_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # ทำให้อยู่ในฟอร์มของ (xซ้าย, yซ้าย, xขวา, yขวา)
    horizontal_lines = [[x1, y1, x2, y2] if (x1 < x2) or (x1 == x2 and y1 < y2) else [x2, y2, x1, y1]
    for [x1, y1, x2, y2] in horizontal_lines]

    # เรียงเส้นจากล่างขึ้นบน
    horizontal_lines.sort(key=lambda line: line[1], reverse=True)

    all_triangles = 0

    base_info = []
    
    # การนับ จะนับเส้น dividing_lines ที่ผ่านเส้น horizontal_lines แต่ละเส้น โดยไม่นับสองฝั่งด้านข้าง (บวกเพิ่มทีหลัง)

    for h in horizontal_lines :
        x1, y1, x2, y2 = h
        y1 -= 5
        y2 -= 5
        x1 += 15
        x2 -= 15
        
        # หาความชันของเส้น h
        mh = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
        
        # หา ch ตามสมการ y = mx + c
        ch = y1 - mh * x1
        
        number_of_dividing = 0
        
        for d in dividing_lines :
            dx1, dy1, dx2, dy2 = d
            
            # หาความชันของเส้น d
            m = (dy2 - dy1) / (dx2 - dx1) if dx2 - dx1 != 0 else float('inf')
            
            # หา cd ตามสมการ y = mx + c
            c = dy1 - (m * dx1)
            
            # y = mx + c --(1)
            # y = mhx + ch --(2)
            
            # นำสองสมการมาลบกัน
            # 0 = x(m - mh) + (c + ch) --((1) - (2))
            # ch - c = x(m - mh)
            # x = (ch - c) / (m - mh) --(3)
            
            if m != mh :
                # หา new_dx ซึ่งก็คือการหา x จากสมการ (3) [x ของจุดที่สองเส้นตัดกัน]
                new_dx = (ch - c) / (m - mh) if m != float('inf') else dx1
                
                # หา new_dy โดยแทน new_dx เข้าไปในสมการ (1)
                new_dy = m * new_dx + c if m != float('inf') else mh * new_dx + ch
                
                # เช็กเงื่อนไข
                # new_dx ต้องอยู่ระหว่างจุดปลายสองจุดของเส้นแนวนอน
                # new_dy ต้องอยู่ระหว่างจุดปลายสองจุดของเส้นแนวนอน
                # ถ้าครบ 2 เงื่อนไข แปลว่า (new_dx, new_dy) คือจุดที่เส้น h และ d ตัดกัน
                if min(x1, x2) <= int(new_dx) <= max(x1, x2) and min(y1, y2) <= int(new_dy) <= max(y1, y2) :
                    number_of_dividing += 1
        
        # บันทึกใน list ว่าเส้น h นี้มีเส้น d ให้เลือกกี่เส้น
        base_info.append(number_of_dividing + 2)

        # บวกเพิ่มในคำตอบรวม
        all_triangles += ((number_of_dividing + 1) * (number_of_dividing + 2)) / 2
    
    return 'triangle', all_triangles, base_info, ret_image

def funcRectangles(image, threshold, approx) :
    vertices = [tuple(point[0]) for point in approx]
    
    sides = [(vertices[i][0], vertices[i][1], vertices[(i + 1) % len(vertices)][0], vertices[(i + 1) % len(vertices)][1])
            for i in range(len(vertices))]
    sides = [(x1, y1, x2, y2) if y1 < y2 or (y1 == y2 and x1 < x2) else (x2, y2, x1, y1)
            for x1, y1, x2, y2 in sides]

    lines = cv2.HoughLinesP(threshold, 2, np.pi / 180, threshold=50, minLineLength=0, maxLineGap=10)
    lines = [
        (x1, y1, x2, y2) if (y1 < y2) or (y1 == y2 and x1 < x2) else (x2, y2, x1, y1)
        for [[x1, y1, x2, y2]] in lines
    ]
    
    lines = sorted(lines, key=lambda line: line_length(line), reverse=True)
    
    horizontal_lines = []
    vertical_lines = []
    
    for side in sides :
        x1, y1, x2, y2 = side
        if abs(y1 - y2) <= 20 :
            horizontal_lines.append(side)
        else :
            vertical_lines.append(side)
    
    # นับและแยกเส้นแนวตั้ง-แนวนอน
    if lines is not None :
        for line1 in lines :
            keep = True
            hor = None
            x1, y1, x2, y2 = line1
            
            if cv2.pointPolygonTest(np.array(approx), (float((x1 + x2) / 2), float((y1 + y2) / 2)), False) >= 0 :
                if abs(y1 - y2) <= 20 :
                    hor = True
                    for line2 in horizontal_lines :
                        if is_horizontally_overlapping(line1, line2) :
                            keep = False
                            break
                else :
                    hor = False
                    for line2 in vertical_lines :
                        if is_vertically_overlapping(line1, line2) :
                            keep = False
                            break
                
                if keep :
                    if hor :
                        horizontal_lines.append(line1)
                    else :
                        vertical_lines.append(line1)

    """
    for l in horizontal_lines :
        show_line(image, l)
    
    for l in vertical_lines :
        show_line(image, l)
    """
    
    # วาดเส้นในภาพ
    ret_image = image.copy()
    for x1, y1, x2, y2 in horizontal_lines :
        cv2.line(ret_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for x1, y1, x2, y2 in vertical_lines :
        cv2.line(ret_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # ทำให้พิกัดเส้นอยู่ในรูป (xซ้าย, yซ้าย, xขวา, yขวา)
    horizontal_lines = [[x1, y1, x2, y2] if (x1 < x2) or (x1 == x2 and y1 < y2) else [x2, y2, x1, y1]
                        for [x1, y1, x2, y2] in horizontal_lines]
    
    # เรียงเส้นล่างขึ้นบน
    horizontal_lines.sort(key=lambda line: line[1], reverse=True)
    
    all_rectangles = 0
    
    # เก็บว่าแต่ละเส้นแนวนอน มีเส้นแนวตั้งผ่านกี่เส้น
    base_info = []
    
    for h in horizontal_lines :
        # บีบเส้นให้แคบเพื่อไม่ให้แตะกับสองเส้นที่โอบด้านข้าง
        x1, y1, x2, y2 = h
        y1 -= 5
        y2 -= 5
        x1 += 15
        x2 -= 15

        number_of_vertical = 0

        for v in vertical_lines :
            vx1, vy1, vx2, vy2 = v

            # เช็กว่า vx อยู่ระหว่าง x1 กับ x2 ของเส้น h หรือเปล่า
            if min(x1, x2) <= vx1 <= max(x1, x2) :
                number_of_vertical += 1
        
        # บันทึกว่าเส้นแนวนอนนั้นเลือกเส้นแนวตั้งได้กี่เส้น
        base_info.append(number_of_vertical + 2)
    
    for i in range(len(horizontal_lines)) :
        for j in range(i+1, len(horizontal_lines)) :
            t = min(base_info[i], base_info[j])
            all_rectangles += t * (t-1) / 2
      
    return 'rectangle', all_rectangles, base_info, ret_image
 
def func1(file_path) :
    try:
        image_original = cv2.imread(file_path)
        image = image_original.copy()
        image_1 = image_original.copy()

        ratio = image.shape[0]/image.shape[1]
        image = cv2.resize(image, (500, int(500 * ratio)))
        image_1 = cv2.resize(image_1, (500, int(500 * ratio)))

        blurred_1 = cv2.GaussianBlur(image_1, (3, 3), 0)
        blurred_1 = cv2.medianBlur(blurred_1, 3)
        gray_1 = cv2.cvtColor(blurred_1, cv2.COLOR_BGR2GRAY)
        threshold_1 = cv2.Canny(gray_1, 50, 150, apertureSize=3)
        threshold_1 = cv2.morphologyEx(threshold_1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

        main_contour_temp = findMainContours(image_1, threshold_1)
        epsilon = 0.02 * cv2.arcLength(main_contour_temp, True)
        approx = cv2.approxPolyDP(main_contour_temp, epsilon, True)
        
        vertices = [tuple(point[0]) for point in approx]
        sides = [(vertices[i][0], vertices[i][1], vertices[(i + 1) % len(vertices)][0], vertices[(i + 1) % len(vertices)][1])
                for i in range(len(vertices))]
        sides = [(x1, y1, x2, y2) if y1 < y2 or (y1 == y2 and x1 < x2) else (x2, y2, x1, y1)
                for x1, y1, x2, y2 in sides]
        
        base = min(sides, key=lambda side: abs(slope(side)))
        if base[0] > base[2] :
            base = (base[2], base[3], base[0], base[1])
            
        rotated = rotate_image(image_1, base)
        
        blurred_1 = cv2.GaussianBlur(rotated, (3, 3), 0)
        blurred_1 = cv2.medianBlur(rotated, 3)
        gray_1 = cv2.cvtColor(blurred_1, cv2.COLOR_BGR2GRAY)
        threshold_1 = cv2.Canny(gray_1, 50, 150, apertureSize=3)
        threshold_1 = cv2.morphologyEx(threshold_1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        
        main_contour_temp = findMainContours(rotated, threshold_1)

        x, y, w, h = cv2.boundingRect(main_contour_temp)
        image = rotated[y-10: y+h+10, x-10: x+w+10]

        ratio = image.shape[0]/image.shape[1]
        image = cv2.resize(image, (256, int(256 * ratio)))
        width, height = image.shape[:2]
        
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        threshold = cv2.Canny(gray, 50, 150, apertureSize=3)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

        main_contour = findMainContours(image, threshold)
        epsilon = 0.02 * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        
        image_type = ''
        answer = 0
        arr_info = []
        ret_image = None

        if len(approx) == 3 :
            image_type, answer, arr_info, ret_image = funcTriangles(image, threshold, approx)
        elif len(approx) == 4 :
            image_type, answer, arr_info, ret_image = funcRectangles(image, threshold, approx)
        else :
            return None
    except :
        return None
    
    return image_type, answer, arr_info, ret_image

def clean() :
    if os.path.exists(UPLOAD_FOLDER) :
        for filename in os.listdir(UPLOAD_FOLDER) :
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            try :
                if os.path.isfile(filepath) :
                    os.remove(filepath)
            except :
                pass

@app.route('/')
def home() :
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file() :
    if 'image' not in request.files :
        return jsonify({"error" : "No file uploaded"}), 400
    
    file = request.files['image']
    if file :
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    
    ret = func1(filepath)
    if ret :
        image_type, answer, arr_info, ret_image = func1(filepath)
    else :
        return {
            'image_type' : None, 
            'answer' : None,
            'arr_info' : None,
            'ret_image_url' : None
        }
    
    name, _ = os.path.splitext(filename)
    new_filename = f'{name}_gen.png'
    filepath = os.path.join(app.config['GENERATED_FOLDER'], new_filename)
    cv2.imwrite(filepath, ret_image)
    timestamp = int(time.time())
    ret_image_url = url_for('static', filename=f'generated/{new_filename}') + f'?v={timestamp}'
    
    result = {
        'image_type' : image_type,
        'answer' : answer,
        'arr_info' : arr_info,
        'ret_image_url' : ret_image_url
    }
    
    del ret_image, image_type, answer, arr_info
    gc.collect()
    
    return result

@app.route('/uploads/<filename>')
def uploaded_file(filename) :
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/topic')
def topic() :
    return render_template('topic.html')

@app.route('/scanner')
def scanner() :
    return render_template('scanner.html')

@app.route('/example')
def example() :
    return render_template('example.html')

if __name__ == '__main__' :       
    app.run(host='0.0.0.0', port=5000, debug=True)           