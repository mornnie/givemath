function setupSideMenu(){
    const menuToggleOutside = document.getElementById('menu-toggle-outside');
    const menuToggleInside = document.getElementById('menu-toggle-inside');
    const sideMenu = document.getElementById('side-menu');
    const overlay = document.getElementById('overlay');

    if(!menuToggleOutside || !menuToggleInside || !sideMenu || !overlay) return;

    function useSideMenu(){
        if(sideMenu.style.width == '250px'){
            sideMenu.style.width = '0';
            overlay.style.display = 'none';
        }
        else{
            sideMenu.style.width = '250px';
            overlay.style.display = 'block';
        }
    }

    menuToggleOutside.addEventListener('click', useSideMenu);
    menuToggleInside.addEventListener('click', useSideMenu);
    overlay.addEventListener('click', useSideMenu);
}

function setupDropDown(){
    const dropdownButton = document.getElementById('dropdown-button');
    const dropdownMenu = document.getElementById('dropdown-menu');

    if(!dropdownButton || !dropdownMenu) return;

    function useDropDown(){
        dropdownButton.classList.toggle('active');
        if(dropdownMenu.style.height == '100px'){
            dropdownMenu.style.height = '0';
            dropdownMenu.style.border = 'none';
        }
        else{
            dropdownMenu.style.height = '100px';
            dropdownMenu.style.display = 'flex';
            dropdownMenu.style.flexDirection = 'column';
            dropdownMenu.style.border = 'black 1px solid';
        } 
    }

    dropdownButton.addEventListener('click', useDropDown);
}

function setupInput(){
    const buttonCamera = document.getElementById('button-camera');
    const realCamera = document.getElementById('real-camera');
    const buttonInput = document.getElementById('button-input');
    const realInput = document.getElementById('real-input');
    const buttonProcess = document.getElementById('button-process');
    const uploadcontainer = document.getElementById('upload-container');
    const textResult = document.getElementById('text-result');
    const textExplain = document.getElementById('text-explain');

    if(buttonCamera && realCamera){
        buttonCamera.addEventListener('click', () => {
            realCamera.click();
        });
    }

    if(realCamera && uploadcontainer){
        realCamera.addEventListener('change', () => {
            const file = realCamera.files[0];

            if(file){
                const img = document.createElement('img');
                img.src = URL.createObjectURL(file)
                img.style.width = '300px';
                img.style.height = '300px';
                uploadcontainer.innerHTML = '';
                uploadcontainer.appendChild(img);
            }
        });
    }

    if(buttonInput){
        buttonInput.addEventListener('click', () => {
            realInput.click();
        });
    }

    if(realInput && uploadcontainer){
        realInput.addEventListener('change', () => {
            const file = realInput.files[0];

            if(file){
                const img = document.createElement('img');
                img.src = URL.createObjectURL(file)
                img.style.width = '300px';
                img.style.height = '300px';
                uploadcontainer.innerHTML = '';
                uploadcontainer.appendChild(img);
            }
        });
    }

    if(!buttonProcess || !realCamera || !uploadcontainer || !textResult || !textExplain) return;
    buttonProcess.addEventListener('click', async() => {
        let file = null;

        if(realCamera.files.length > 0){
            file = realCamera.files[0];
        }
        else{
            file = realInput.files[0];
        }

        if(file){
            const formData = new FormData();
            formData.append('image', file);
            try{
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if(!data.image_type) throw new Error();
                const img = document.createElement('img');
                img.src = data.ret_image_url;
                img.style.width = '300px';
                img.style.height = '300px';
                
                uploadcontainer.innerHTML = '';
                uploadcontainer.appendChild(img);

                const resultTemplate = "คำตอบ คือ {{answer}} รูป<br><br>";
                const explanationTemplateTri = `เมื่อเลือกเส้นแนวนอนเส้นที่ {{i}} และเลือกเส้นด้านประกอบมุมยอด 2 เส้น จากทั้งหมด {{x}} เส้น สร้างได้ \\( \\Large \\binom{{x}}{2} \\) = {{y}}  รูป<br><br>`;
                const explanationTemplateRec = `เมื่อเลือกเส้นแนวนอนเส้นที่ {{i}} สร้างได้ <br> {{subtemplate}} = {{sum}} รูป`;
                const explanationTemplateRec2 = `\\( \\Large \\binom{{x}}{2} \\) `;
                function renderTemplate(template, values){
                    return template.replace(/{{(.*?)}}/g, (match, key) => values[key.trim()] ?? '');
                }

                let resultHTML = '';
                let explanationHTML = '';

                resultHTML += renderTemplate(resultTemplate, {answer: data.answer});

                if(data.image_type === 'triangle'){
                    explanationHTML += 'พิจารณาเส้นแนวนอนจากเส้นล่างสุดขึ้นไปยังเส้นบนสุด<br><br>';
                    data.arr_info.forEach((x, index) => {
                        const i = index + 1;
                        const y = x * (x-1) / 2;
                        explanationHTML += renderTemplate(explanationTemplateTri, { i, x, y });
                    })
                }
                else if(data.image_type === 'rectangle'){
                    explanationHTML += 'พิจารณาเส้นแนวนอนจากเส้นล่างสุดขึ้นไปยังเส้นบนสุด<br><br>'
                    
                    for(let i=0; i<data.arr_info.length; i++){
                        let sum = 0;
                        let explanationTemp = '';

                        for(let j=i+1; j<data.arr_info.length; j++){
                            const x = Math.min(data.arr_info[i], data.arr_info[j]);
                            sum += x * (x-1) / 2;
                            explanationTemp += renderTemplate(explanationTemplateRec2, { x });
                            if(j != data.arr_info.length - 1) explanationTemp += '+ ';
                        }
                        explanationHTML += renderTemplate(explanationTemplateRec, { i: i+1, subtemplate: explanationTemp, sum });
                        if(i != 0) explanationHTML += ' (ไม่นับรูปซ้ำ)<br><br>';
                        else explanationHTML += '<br><br>'
                    }
                }

                textResult.innerHTML = resultHTML;
                textExplain.innerHTML = explanationHTML;

                textResult.style.color = 'black';
                textExplain.style.color = 'black'; 

                if(window.MathJax){
                    MathJax.typeset();
                }
            }
            catch{
                textResult.innerHTML = 'ไม่สามารถหาคำตอบได้'
                textExplain.innerHTML = 'ขออภัย ปัญหานี้อาจอยู่นอกขอบเขตที่เราแก้ได้'
                
                textResult.style.color = 'red';
                textExplain.style.color = 'red';
            }
        }
    });
}

function renderMarkdown(id, markdown){
    const obj = document.getElementById(id);
    if(!obj) return Promise.resolve();

    const html = marked.parse(markdown);
    obj.innerHTML = html;

    return MathJax.typesetPromise();
}

const topicMarkdown = `
**การจัดหมู่** คือ การจัดสิ่งของที่แตกต่างกันออกเป็นหมู่<br>
โดยไม่ยึดถืออันดับเป็นสำคัญ<br><br>
เช่น ต้องการเลือกตัวอักษร 2 ตัว จาก {A, B, C, D}<br>
การเลือก AB และการเลือก BA นั้นเป็นวิธีเดียวกัน<br>
เพราะลำดับไม่สำคัญ การสลับที่จึงไม่มีความหมาย<br><br>
จำนวนวิธีจัดหมู่ของ n สิ่งที่ต่างกัน โดยเลือกมา r สิ่ง<br>จะเท่ากับ<br>
<div style="text-align: center">
$ C_{n,r} $ = $ \\Large \\binom{n}{r} $ = $ \\Large \\frac{n!}{(n-r)!r!} $
</div>

**ตัวอย่าง** ต้องการเลือกกรรมการนักเรียนห้องละ 2 คน<br>
โดยมีผู้สมัครจากห้อง A ทั้งหมด 3 คน และผู้สมัครจากห้อง B ทั้งหมด 5 คน จะสามารถเลือกได้กี่วิธี<br><br>
**แนวคิด**<br>
เลือกผู้สมัครจากห้อง A ได้ 2 จาก 3 คน คือ $ \\Large \\binom{3}{2} $ <br>
เลือกผู้สมัครจากห้อง B ได้ 2 จาก 5 คน คือ $ \\Large \\binom{5}{2} $ <br>
จากกฎการคูณ ได้ว่า จำนวนวิธีทั้งหมดเท่ากับ <br><br>
$ \\Large \\binom{3}{2} $ x $ \\Large \\binom{5}{2} $ <br><br>
= $ \\Large \\frac{3!}{(3-2)!2!} $ x $ \\Large \\frac{5!}{(5-2)!2!} $ <br><br>
= $ \\Large \\frac{3!}{2!} $ x $ \\Large \\frac{5!}{3!2!} $ <br><br>
= 3 x 10 <br><br>
= 30 วิธี <br><br>
**ตอบ** 30 วิธี <hr>

### การจัดหมู่กับโจทย์การนับรูปภาพ<br>
การจัดหมู่สามารถใช้แก้โจทย์ประเภทที่ให้นับรูปสามเหลี่ยม / สี่เหลี่ยมทั้งหมดในภาพได้เช่นกัน <br> ดังตัวอย่างต่อไปนี้

**ตัวอย่าง** จงหาจำนวนรูปสี่เหลี่ยมมุมฉากทั้งหมด<br><br>
<img src='static/topic-problem.png' height='300px'><br>
**แนวคิด**<br><br>
ในการสร้างสี่เหลี่ยมมุมฉาก 1 รูป<br><br>
ต้องใช้เส้นแนวตั้ง 2 เส้น และเส้นแนวนอน 2 เส้น<br><br>
เลือกเส้นแนวตั้งได้ 2 เส้น จาก 4 เส้น $ C_{4,2} $ <br><br>
เลือกเส้นแนวนอนได้ 2 เส้น จาก 5 เส้น $ C_{5,2} $ <br><br>
ดังนั้น จำนวนรูปสี่เหลี่ยมมุมฉากทั้งหมด <br><br>
= $ C_{4,2} $ x $ C_{5,2} $ = 6 x 10 = 60 รูป<br><br>
**ตอบ** 60 รูป
`;

const exampleMarkdown = `
### จงหาจำนวนรูปสามเหลี่ยมทั้งหมด
<img src='static/example_tri.jpg' height='300px'>

### แนวคิด
สามเหลี่ยม ประกอบด้วย ด้านฐาน 1 ด้าน และด้านประกอบมุมยอด 2 ด้าน<br><br>
ให้เส้นแนวนอนแต่ละเส้นเป็นฐานของรูปสามเหลี่ยมย่อย<br><br>
เมื่อพิจารณาเส้นแนวนอนจากด้านล่างขึ้นด้านบน<br><br>
เมื่อเลือกเส้นแนวนอนที่ 1 จะเลือกด้านประกอบมุมยอดได้ 2 จาก 4 เส้น<br>
เท่ากับสร้างได้ $ \\Large \\binom{4}{2} $ = $ \\Large \\frac{4!}{2!2!} $ = 6 รูป<br><br>
เมื่อเลือกเส้นแนวนอนที่ 2 จะเลือกด้านประกอบมุมยอดได้ 2 จาก 4 เส้น<br>
เท่ากับสร้างได้ $ \\Large \\binom{4}{2} $ = $ \\Large \\frac{4!}{2!2!} $ = 6 รูป<br><br>
เมื่อเลือกเส้นแนวนอนที่ 3 จะเลือกด้านประกอบมุมยอดได้ 2 จาก 4 เส้น<br>
เท่ากับสร้างได้ $ \\Large \\binom{4}{2} $ = $ \\Large \\frac{4!}{2!2!} $ = 6 รูป<br><br>
รวม สร้างสามเหลี่ยมได้ 6 + 6 + 6 = 18 รูป<br><br>
**ตอบ** 30 รูป<hr>

### จงหาจำนวนรูปสี่เหลี่ยมมุมฉากทั้งหมด
<img src='static/example_rec.jpg' height='300px'>

### แนวคิด
ในการสร้างสี่เหลี่ยมมุมฉากหนึ่งรูป<br><br>
ต้องใช้เส้นแนวตั้ง 2 เส้น และเส้นแนวนอน 2 เส้น<br><br>
เลือกเส้นแนวตั้งได้ 2 เส้น จาก 5 เส้น<br><br>
เลือกเส้นแนวนอนได้ 2 เส้น จาก 3 เส้น<br><br>
ดังนั้น สร้างรูปสี่เหลี่ยมมุมฉากได้ทั้งหมด<br><br>
$ \\Large \\binom{5}{2}\\binom{3}{2} $ = $ \\Large \\frac{5!}{2!3!} $ x $ \\Large \\frac{3!}{2!1!} $ = 30 รูป<br><br>
**ตอบ** 30 รูป
`;

document.addEventListener('DOMContentLoaded', () => {
    setupSideMenu();
    setupDropDown();
    setupInput();

    renderMarkdown('paragraph', topicMarkdown);
    renderMarkdown('triangle-text', exampleMarkdown);
});
