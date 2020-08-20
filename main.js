
const errorHandler = setupErrorHandler();

window.addEventListener('load', () => run().catch((error) => errorHandler.onerror(error)));

async function run()
{
    const loadingEl = document.getElementById("loading");
    const mainContainer = document.getElementById("mainContainer");
    const data = new MnistData();
    await data.load();
    loadingEl.parentElement.removeChild(loadingEl);
    mainContainer.style.visibility = 'visible';

    const HIDDEN_NODES = 8;
    const model = tf.sequential({
        layers: [
            tf.layers.dense({inputShape: [IMAGE_SIZE], units: HIDDEN_NODES, activation: 'relu', name: 'layer1'}),
            tf.layers.dense({units: 10, activation: 'softmax', name: 'layer2'}),
        ]
    });

    model.compile({
       loss: 'meanSquaredError',
       optimizer: 'sgd'
    });

    const trainData = data.getTrainData();
    const testData = data.getTestData();
    const testXS = testData.xs;
    const testYS = testData.ys.dataSync();
    const trainLabels = [];
    for (let i=0; i<testYS.length; i+=10)  trainLabels[i/10] = maxIndex(testYS, i, i+10) - i;

    const panel = new NetworkPanel(model, testXS, trainLabels);

    let epoch = 0;
    model.fit(trainData.xs, trainData.ys, {epochs: 1000, callbacks: {onEpochEnd: () => {
        const n = testXS.shape[0];
        const result = model.predict(testXS).dataSync();
        const resultLabels = [];
        let success = 0;
        for (let i=0; i<n; i++) {
            const res = maxIndex(result, i*10, (i+1)*10) - i*10;
            const exp = trainLabels[i];
            if (res === exp)  success++;
            resultLabels[i] = res;
        }

        panel.epochEnd(++epoch, success / n, resultLabels);
    }}});

    const colorSwitch = document.getElementById('colorSwitch');
    panel.setColorMode(colorSwitch.checked);
    colorSwitch.addEventListener('change', (e) => panel.setColorMode(e.target.checked));

    const sortOutputsSwitch = document.getElementById('sortOutputsSwitch');
    panel.setSortOutputs(sortOutputsSwitch.checked);
    sortOutputsSwitch.addEventListener('change', (e) => panel.setSortOutputs(e.target.checked));
}

class NetworkPanel
{
    constructor(model, inputData, inputLabels) {
        this.model = model;
        this.inputData = inputData;
        this.inputLabels = inputLabels;
        this.innerLayerColumn1Sep = ": ";
        this.innerLayerColumn2Sep = "  ";
        this.innerLayerWeightPrecision = 5;

        this.calculateModelData();
        this.initOutputLayer();
        this.initInputLayer('inputLayer', inputLabels);
        this.initInnerLayer('innerLayer', model.layers[0].outputShape[1]);
        this.renderInnerLayer();
    }

    epochEnd(epoch, success, predictedLabels)
    {
        this.predictedLabels = predictedLabels;
        this.calculateModelData();
        this.renderInputLayer();
        this.renderInnerLayer();
        this.renderOutputLayer();

        document.getElementById('resultLabel').innerText = 'epoch: ' + epoch + ', result: ' + Math.round(success * 100) + '%';
    }

    initInputLayer(containerId, inputLabels)
    {
        const indices = Array.from(Array(inputLabels.length).keys());
        shuffle(indices);
        const indicesByNumber = this.inputDataIndicesByNumber = [];
        for (let i=0; i<10; i++)  indicesByNumber.push([]);
        for (let i=0; i<indices.length; i++) {
            const index = indices[i];
            const number = inputLabels[index];
            indicesByNumber[number].push(index);
        }
        const maxNumberIndicesSize = Math.max(...indicesByNumber.map((it) => it.length));

        const marginInParent = 8;
        const arrowWidth = IMAGE_W;
        const scale = 1;
        const margin = 2;
        const arrowPadding = 4;

        const that = this;
        that.selectedIndex = -1;
        that.imageIndexOffset = 0;

        function resize()
        {
            const parentContainer = document.getElementById(containerId);
            const parentWidth = getInnerDimensions(parentContainer).width;
            const width = Math.floor((parentWidth - marginInParent * 11 - arrowWidth * 2) / 10);
            const countPerNumber = that.inputLayerImagesPerNumber = Math.max(1, Math.floor((width / scale + margin) / (IMAGE_W + margin)));
            if (!parentContainer.children[0])  appendArrow(parentContainer, -1, false);
            for (let i=0; i<10; i++)
            {
                let container = parentContainer.children[i+1];
                if (!container)  {
                    container = document.createElement('div');
                    container.style.marginLeft = countPerNumber > 1 ? marginInParent + "px" : '0';
                    parentContainer.appendChild(container);
                }

                while (container.children.length < countPerNumber) {
                    const j = container.children.length;
                    const canvas = document.createElement('canvas');
                    container.appendChild(canvas);
                    canvas.width = IMAGE_W;
                    canvas.height = IMAGE_H;
                    canvas.style.width = Math.round(scale * IMAGE_W) + 'px';
                    canvas.style.height = Math.round(scale * IMAGE_H) + 'px';
                    canvas.style.imageRendering = "pixelated";
                    if (j !== 0)  canvas.style.marginLeft = margin + "px";
                    canvas.onclick = () => {
                        const index = indicesByNumber[i][j + that.imageIndexOffset];
                        const lastIndex = that.selectedIndex;
                        if (lastIndex !== index) {
                            if (lastIndex !== -1 && lastIndex === that.selectedCanvas.index)  that.renderInputImage(that.selectedCanvas, lastIndex, false);
                            that.renderInputImage(canvas, index, true);
                            that.selectedIndex = index;
                            that.selectedData = canvas.inputData;
                            that.selectedCanvas = canvas;
                        } else {
                            that.renderInputImage(canvas, index, false);
                            that.selectedIndex = -1;
                            that.selectedData = null;
                            that.selectedCanvas = null;
                        }
                        that.calculateModelData();
                        that.renderInnerLayer();
                        that.renderOutputLayer();
                    }
                }
                while (container.children.length > countPerNumber)  container.removeChild(container.children[container.children.length-1]);
            }
            if (!parentContainer.children[11])  appendArrow(parentContainer, 1, countPerNumber> 1);

            that.renderInputLayer();
        }

        function appendArrow(parent, direction, marginLeft)
        {
            const arrow = document.createElement('div');
            arrow.innerHTML = direction > 0 ? "⮞" : "⮜";
            arrow.className = "arrow";
            arrow.style.width = arrowWidth + 'px';
            arrow.style.height = Math.round(scale * IMAGE_H - arrowPadding) + 'px';
            arrow.style.paddingTop = arrowPadding + 'px';
            if (marginLeft)  arrow.style.marginLeft = marginInParent + "px";
            arrow.onclick = () => {
                that.imageIndexOffset = mod(that.imageIndexOffset + direction * that.inputLayerImagesPerNumber, maxNumberIndicesSize);
                that.renderInputLayer();
            };
            parent.appendChild(arrow);
        }

        this.addOnResize(resize);
    }

    renderInputLayer()
    {
        const parentContainer = document.getElementById('inputLayer');
        for (let j=0; j<10; j++)  {
            const container = parentContainer.children[j+1];
            for (let i=0; i<this.inputLayerImagesPerNumber; i++)  {
                const canvas = container.children[i];
                const index = this.inputDataIndicesByNumber[j][i + this.imageIndexOffset];
                this.renderInputImage(canvas, index, index === this.selectedIndex)
            }
        }
    }

    renderInputImage(canvas, index, selected)
    {
        if (index == null) {
            canvas.style.visibility = "hidden";
            return;
        }
        else  canvas.style.visibility = "visible";

        const success = this.predictedLabels && this.predictedLabels[index] === this.inputLabels[index];

        const x = this.inputData.slice([index,0], [1,IMAGE_SIZE]);
        canvas.inputData = x;
        canvas.index = index;
        const source = x.dataSync();
        if (source.length !== IMAGE_SIZE) throw "Wrong input vector size";

        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(IMAGE_W, IMAGE_H);
        const data = imgData.data;
        for (let i=0; i<data.length; i+=4) {
            let c1 = Math.round(source[i/4] * 255);
            let c2 = Math.round(source[i/4] * 255 * 0.6);
            data[i  ] = success ? c2 : c1;
            data[i+1] = success ? c1 : c2;
            data[i+2] = selected ? 255 : c2;
            data[i+3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
    }

    initInnerLayer(containerId, n)
    {
        const fontMetricsBySize = [8,9,10,11,12,13,14].map((it) => getFontMetrics("monospace", it));

        const that = this;
        const relativeMarginX = 3;
        const relativeMarginY = 3;
        const charactersPerNumber = 2 + this.innerLayerWeightPrecision;
        const charactersPerRow = 1 + this.innerLayerColumn1Sep.length + charactersPerNumber * 2 + this.innerLayerColumn2Sep.length;
        const col1Sep = this.innerLayerColumn1Sep.replace(/ /g, "&nbsp;");
        const col2Sep = this.innerLayerColumn2Sep.replace(/ /g, "&nbsp;");

        function resize()
        {
            const container = document.getElementById(containerId);
            const {width,height} = getInnerDimensions(container);

            let rowCount;
            let fontMetrics;
            let scale = 1;
            //  find the initial scale value that gives an image large than the text with minimal width
            for (;; scale++)  {
                const imageWidth = IMAGE_W * scale;
                const rowWidth = charactersPerRow * fontMetricsBySize[0].width;
                if (rowWidth <= imageWidth)  break;
            }
            //  increase font size and scale until the content fits the page
            for (let f=0;;)  {
                let s = scale;
                if (f + 1 < fontMetricsBySize.length && IMAGE_W * s >= charactersPerRow * fontMetricsBySize[f + 1].width)  f++;
                else  s++;
                const imageWidth = IMAGE_W * s;
                const imageHeight = IMAGE_H * s;
                const marginX = relativeMarginX * s;
                const marginY = relativeMarginY * s;
                const fontMetrics_ = fontMetricsBySize[f];
                let countInRow = Math.max(1, Math.floor((width + marginX) / (imageWidth + marginX)));
                let rowCount_ = Math.ceil(n / countInRow);
                let rowHeight = imageHeight + fontMetrics_.height * 12;
                let contentHeight = rowHeight * rowCount_ + marginY * (rowCount_);
                if (s !== scale && contentHeight > height)  break;
                rowCount = rowCount_;
                scale = s;
                fontMetrics = fontMetrics_;
            }

            //  create missing elements
            let items = container.items;
            if (container.childElementCount !== rowCount)  {
                if (items == null) {
                    items = [];

                    for (let i=0; i<n; i++) {
                        const el = document.createElement('div');
                        items.push(el);
                        const canvas = el.canvas = document.createElement('canvas');
                        canvas.width = IMAGE_W;
                        canvas.height = IMAGE_H;
                        el.appendChild(canvas);
                        el.biasLabel = document.createElement('div');
                        el.appendChild(el.biasLabel);
                        el.outputLabel = document.createElement('div');
                        el.appendChild(el.outputLabel);
                        el.weightRows = [];
                        for (let j=0; j<10; j++) {
                            const row = document.createElement('div');
                            el.appendChild(row);
                            el.weightRows.push(row);
                            row.numberLabel = that.createSpan(row, '');
                            that.createSpan(row, col1Sep);
                            row.weightLabel = that.createSpan(row, '');
                            that.createSpan(row, col2Sep);
                            row.outputLabel = that.createSpan(row, '');
                            row.onmouseover = (e) => {
                                e.stopPropagation();
                                that.highlightOutputNumberWeight(j, i);
                            }
                        }
                        el.onmouseover = (e) => that.highlightOutputInnerNode(el, i);
                        el.onmouseleave = (e) => that.removeHighlight();
                    }
                }
                for (let r=0; r<container.children.length; r++)  {
                    const row = container.children[r];
                    for (let i=0; i<row.children.length; i++)  row.removeChild(row.children[i]);
                }
                while (container.children.length < rowCount)  container.appendChild(document.createElement('div'));
                while (container.children.length > rowCount)  container.removeChild(container.children[container.children.length-1]);
                const rowSize = Math.ceil(n / rowCount);
                for (let r=0, j=0; r<rowCount; r++)  for (let i=0; i<rowSize && j<items.length; i++, j++)  {
                    container.children[r].appendChild(items[j]);
                }
                container.items = items;
            }
            for (let i=0; i<container.children.length; i++)  container.children[i].style.marginTop = (relativeMarginY * scale) + "px";
            for (let i=0; i<n; i++) {
                const canvas = items[i].canvas;
                canvas.style.width = scale * IMAGE_W + 'px';
                canvas.style.height = scale * IMAGE_H + 'px';
                items[i].style.fontSize = fontMetrics.size + 'px';
            }
        }

        this.addOnResize(resize)
    }

    renderInnerLayer() {
        const container = document.getElementById('innerLayer');
        const n = this.innerNodeCount;
        const data = this.modelData;
        const weights = data.layer1Weights;
        const biases = data.layer1Biases;
        const layer2Weights = data.layer2Weights;
        const output = data.innerResults;
        const maxOutput = data.maxWeight;

        for (let j=0; j<n; j++) {
            const div = container.items[j];
            div.biasLabel.innerHTML = "bias: &nbsp;&nbsp;" + this.formatWeight(biases[j]);
            div.outputLabel.innerHTML = "output:&nbsp;" + this.formatWeight(output[j]);
            const k = j * 10;
            let outValues = Array.from(layer2Weights.slice(k, k + 10)).map((v,i) => ({index: i, value: v}));
            if (this.sortOutputs)  outValues = outValues.sort((a,b) => Math.abs(b.value) - Math.abs(a.value));
            for (let i=0; i<10; i++)  {
                const item = outValues[i];
                const weightedOutput = output[j] * item.value;
                div.weightRows[i].numberLabel.innerText = item.index.toString();
                div.weightRows[i].weightLabel.innerText = this.formatWeight(item.value);
                div.weightRows[i].outputLabel.innerText = isNaN(weightedOutput) ? '' : this.formatWeight(weightedOutput);
                div.weightRows[i].outputLabel.style = this.barStyle(weightedOutput / maxOutput);
            }
            this.renderWeights(div.canvas, weights, j, n);
        }
    }

    renderWeights(canvas, weights, j, n) {
        const selectedData = this.selectedData && this.selectedData.dataSync();
        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(IMAGE_W, IMAGE_H);
        const data = imgData.data;
        for (let i=0; i<data.length; i+=4) {
            const w = weights[j + i/4 * n] * 4;
            let r,g,b;
            if (this.colored) {
                g = Math.round(Math.max(0,  w) * 255);
                r = Math.round(Math.max(0, -w) * 255);
                b = 0;
            }
            else  r = g = b = Math.round(clamp((w+1)/2, 0, 1) * 255);
            let sel = 0.4 * (selectedData ? 1 - selectedData[i/4] : 0);
            data[i  ] = Math.round(r * (1-sel) +   0 * sel);
            data[i+1] = Math.round(g * (1-sel) +   0 * sel);
            data[i+2] = Math.round(b * (1-sel) + 255 * sel);
            data[i+3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
    }

    initOutputLayer()
    {
        const that = this;
        const innerLayerContainer = document.getElementById('innerLayer');
        const container = document.getElementById('outputLayer');

        if (container.childElementCount === 0)
        {
            const n = this.model.layers[0].trainableWeights[0].shape[1];

            for (let i=0; i<10; i++) {
                let div = document.createElement('div');
                const that = this;
                function addRow(label) {
                    const row = document.createElement('div');
                    that.createSpan(row, label);
                    row.valueSpan = that.createSpan(row, '');
                    div.appendChild(row);
                    return row;
                }

                div.numberLabel = document.createElement('div');
                div.numberLabel.className = 'number';
                div.numberLabel.innerText = i.toString();
                div.appendChild(div.numberLabel);
                div.resultRow = addRow('result:&nbsp;');
                div.resultRow.style = 'font-weight: bold';
                div.biasRow = addRow('bias:&nbsp;&nbsp;&nbsp;');
                div.weightRows = [];
                for (let j=0; j<n; j++) {
                    const ind = j+1;  //this.formatSubscript(j+1);
                    div.weightRows[j] = addRow('w' + ind + '*x' + ind + ':' + (ind < 10 ? '&nbsp;&nbsp;' : ''));
                    div.weightRows[j].onmouseover = (e) => {
                        e.stopPropagation();
                        that.highlightOutputNumberWeight(i, j);
                    }
                }
                div.onmouseover = () => that.highlightOutputNumber(div, i);
                div.onmouseleave = () => that.removeHighlight();
                container.appendChild(div);
            }
        }

        this.renderOutputLayer();
    }

    highlightOutputNumber(div, number)
    {
        const innerContainer = document.getElementById('innerLayer');
        const outerContainer = document.getElementById('outputLayer');
        const innerNodes = innerContainer.items;
        innerContainer.className = 'highlight';
        outerContainer.className = 'highlight';

        const lastEls = document.getElementsByClassName('highlighted');
        for (let i=lastEls.length-1; i>=0; i--)  lastEls[i].className = '';

        div.className = 'highlighted';
        for (let i=0; i<innerNodes.length; i++)  innerNodes[i].weightRows[number].className = 'highlighted';
    }

    highlightOutputInnerNode(div, index)
    {
        const innerContainer = document.getElementById('innerLayer');
        const outerContainer = document.getElementById('outputLayer');
        innerContainer.className = 'highlight';
        outerContainer.className = 'highlight';

        const lastEls = document.getElementsByClassName('highlighted');
        for (let i=lastEls.length-1; i>=0; i--)  lastEls[i].className = '';

        div.className = 'highlighted';
        for (let i=0; i<10; i++)  outerContainer.children[i].weightRows[index].className = 'highlighted';
    }

    highlightOutputNumberWeight(number, weight)
    {
        const innerContainer = document.getElementById('innerLayer');
        const outerContainer = document.getElementById('outputLayer');
        const innerNodes = innerContainer.items;
        innerContainer.className = 'highlight';
        outerContainer.className = 'highlight';

        const lastEls = document.getElementsByClassName('highlighted');
        for (let i=lastEls.length-1; i>=0; i--)  lastEls[i].className = '';

        outerContainer.children[number].weightRows[weight].className = 'highlighted';
        innerNodes[weight].weightRows[number].className = 'highlighted';
    }

    removeHighlight()
    {
        document.getElementById('innerLayer').className = '';
        document.getElementById('outputLayer').className = '';
    }

    renderOutputLayer()
    {
        const container = document.getElementById('outputLayer');
        
        const n = this.innerNodeCount;
        const data = this.modelData;
        const results = data.results;
        const resultIndex = data.resultIndex;
        const resultSum = data.resultSum;
        const layer2Biases = data.layer2Biases;
        const layer2Weights = data.layer2Weights;
        const maxWeight = data.maxWeight;
        const innerResults = data.innerResults;

        for (let i=0; i<10; i++) {
            const div = container.children[i];
            div.numberLabel.className = i === resultIndex ? 'number winner' : 'number';
            div.resultRow.valueSpan.innerHTML = this.formatWeight(results[i]);
            div.resultRow.valueSpan.style = this.barStyle(results[i] / resultSum);
            div.biasRow.valueSpan.innerHTML = this.formatWeight(layer2Biases[i]);
            div.biasRow.valueSpan.style = this.barStyle(layer2Biases[i] / maxWeight);
            //let weightedInput = layer2Biases[i];
            for (let j=0; j<n; j++) {
                const x = innerResults[j] * layer2Weights[j * 10 + i];
                div.weightRows[j].valueSpan.innerHTML = this.formatWeight(x);
                div.weightRows[j].valueSpan.style = this.barStyle(x / maxWeight);
                //weightedInput += x
            }
        }
    }

    calculateModelData() {
        const model = this.model;
        const n = this.innerNodeCount = model.layers[0].trainableWeights[0].shape[1];
        const selectedData = this.selectedData;
        const data = this.modelData = {
            results: [],
            innerResults: []
        };

        const layer1WeightsTensor = model.layers[0].trainableWeights[0];
        const layer1BiasesTensor = model.layers[0].trainableWeights[1];
        if (layer1WeightsTensor.shape[0] !== IMAGE_SIZE)  throw "Wrong layer1 size";
        data.layer1Weights = layer1WeightsTensor.val.dataSync();
        if (data.layer1Weights.length !== IMAGE_SIZE * n)  throw "Wrong layer1 weights data size";
        data.layer1Biases = layer1BiasesTensor.val.dataSync();
        if (data.layer1Biases.length !== n)  throw "Wrong layer1 biases data size";

        const layer2WeightsTensor = model.layers[1].trainableWeights[0];
        const layer2BiasesTensor = model.layers[1].trainableWeights[1];
        if (layer2WeightsTensor.shape[0] !== n || layer2WeightsTensor.shape[1] !== 10)  throw "Wrong output layer weights size";
        if (layer2BiasesTensor.shape[0] !== 10)  throw "Wrong output layer biases size";
        data.layer2Weights = layer2WeightsTensor.val.dataSync();
        data.layer2Biases = layer2BiasesTensor.val.dataSync();

        if (selectedData) {
            const innerModel = tf.model({inputs: this.model.layers[0].input, outputs: this.model.layers[0].output});
            const innerResults = data.innerResults = innerModel.predict(selectedData).dataSync();

            data.results = model.predict(selectedData).dataSync();
            data.resultIndex = maxIndex(data.results, 0, data.results.length);
            data.resultSum = data.results.reduce((a,b) => a+b, 0);
            let maxWeight = data.resultSum;
            for (let i=0; i<10; i++)  Math.max(data.resultSum, Math.abs(data.layer2Biases[i]));
            for (let i=0; i<10; i++)  for (let j=0; j<n; j++)  maxWeight = Math.max(maxWeight, Math.abs(innerResults[j] * data.layer2Weights[j * 10 + i]));
            data.maxWeight = maxWeight;
        }
    }

    addOnResize(func)
    {
        window.addEventListener('resize', _.debounce(func, 300));
        func();
    }

    barStyle(x) {
        if (isNaN(x))  return '';
        const p = Math.abs(Math.round(x * 100));
        const col = x > 0 ? '#8f8' : '#f88';
        return "background: linear-gradient(90deg, "+col+" "+p+"%, #fff "+p+"%)";
    }

    createSpan(parent, text)  {
        const span = document.createElement('span');
        span.innerHTML = text;
        parent.appendChild(span);
        return span;
    }

    formatWeight(x) {
        if (x == null || isNaN(x))  return '&nbsp;'.repeat(this.innerLayerWeightPrecision + 2);
        const a = Math.abs(x);
        return (x < 0 ? "−" : " ") + a.toFixed(this.innerLayerWeightPrecision - (a > 1 ? Math.ceil(Math.log10(a)) : 1));
    }

    formatSubscript(x) {
        if (isNaN(x))  return '';
        const s = x.toString();
        const d = '₁'.charCodeAt(0) - '1'.charCodeAt(0);
        let res = '';
        for (let i=0; i<s.length; i++) {
            res += String.fromCharCode(s.charCodeAt(i) + d);
        }
        return res;
    }

    setColorMode(colored) {
        this.colored = colored;
        this.renderInnerLayer();
    }

    setSortOutputs(sortOutputs) {
        this.sortOutputs = sortOutputs;
        this.renderInnerLayer();
    }
}

function clamp(x, min, max)  {  return x > max ? max : x < min ? min : x;  }

function maxIndex(array, begin, end)
{
    let index = begin;
    let max = array[begin];
    for (let i=begin; i<end; i++)  if (array[i] > max)  {
        index = i;
        max = array[i];
    }
    return index;
}

function shuffle(a) {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

function mod(a,b) {
    return ((a % b) + b) % b;
}


function getFontMetrics(fontFamily, fontSize)
{
    const content = "1234567890";
    let span = document.createElement('span');
    span.style.fontFamily = fontFamily;
    span.style.fontSize = fontSize + "px";
    span.textContent = content;
    document.body.appendChild(span);
    let result = span.getBoundingClientRect();
    document.body.removeChild(span);
    result.width /= content.length;
    result.size = fontSize;
    return result;
}

function getInnerDimensions(element)
{
    const style = getComputedStyle(element);
    return {width: parseInt(style.width), height: parseInt(style.height)};
}

function setupErrorHandler()
{
    const debug = true;
    window.onerror = function (error) {
        console.log('error');
        onerror(error);
    };

    let errorCount = 0;
    let decErrorCountId = null;
    function onerror(error, clientMessage)
    {
        let maxErrors = debug ? 10 : 3;
        if (errorCount + 1 <= maxErrors)  {
            errorCount++;
            const message = error ? (error.message || error.toString()) : 'Error';
            var toastrMessage = debug ? message : (clientMessage || "JavaScript error :( You can try to refresh the page or come back later");
            toastr.error(toastrMessage);
        }
        if (decErrorCountId == null)  decErrorCount();
    }
    function decErrorCount() {
        decErrorCountId = setTimeout(() => {  if (--errorCount > 0)  decErrorCount();  else  decErrorCountId = null;  }, 1000)
    }

    return {onerror: onerror}
}