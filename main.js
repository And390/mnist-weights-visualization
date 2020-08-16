
window.addEventListener('load', () => run());

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
    let xs = trainData.xs;
    xs = xs.reshape([xs.shape[0], IMAGE_SIZE]);
    const ys = trainData.labels;
    let testXS = testData.xs;
    testXS = testXS.reshape([testXS.shape[0], IMAGE_SIZE]);
    const testYS = testData.labels;

    const panel = new NetworkPanel(model, xs, ys);

    let epoch = 0;
    model.fit(xs, ys, {epochs: 1000, callbacks: {onEpochEnd: () => {
        panel.epochEnd();

        const n = testXS.shape[0];
        const result = model.predict(testXS).dataSync();
        const expected = testYS.dataSync();
        let success = 0;
        for (let i=0; i<n; i++) {
            const res = maxIndex(result, i*10, (i+1)*10);
            const exp = maxIndex(expected, i*10, (i+1)*10);
            if (res === exp)  success++;
        }
        document.getElementById('resultLabel').innerText = 'epoch: ' + (++epoch) + ', result: ' + Math.round(success / n * 100) + '%';
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
    constructor(model, trainData, labels) {
        this.model = model;
        this.inputData = trainData;
        this.innerLayerColumn1Sep = ": ";
        this.innerLayerColumn2Sep = "  ";
        this.innerLayerWeightPrecision = 5;

        this.initInputLayer('inputLayer', trainData, labels);
        this.initInnerLayer('innerLayer', model.layers[0].outputShape[1]);
        this.initOutputLayer();
        this.redrawInnerLayer();
    }

    epochEnd()
    {
        this.redrawInnerLayer();
        this.redrawOutputLayer();
    }

    initInputLayer(containerId, inputData, inputDataLabels)
    {
        const indices = Array.from(Array(inputDataLabels.size / 10).keys());
        shuffle(indices);
        const indicesByNumber = [];
        for (let i=0; i<10; i++)  indicesByNumber.push([]);
        const labels = inputDataLabels.dataSync();
        for (let i=0; i<indices.length; i++) {
            const index = indices[i];
            const number = maxIndex(labels, index*10, (index+1)*10) - index*10;
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
        let countPerNumber = -1;
        let imageIndexOffset = 0;

        function resize()
        {
            const parentContainer = document.getElementById(containerId);
            const parentWidth = getInnerDimensions(parentContainer).width;
            const width = Math.floor((parentWidth - marginInParent * 11 - arrowWidth * 2) / 10);
            countPerNumber = Math.max(1, Math.floor((width / scale + margin) / (IMAGE_W + margin)));
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
                        const index = indicesByNumber[i][j + imageIndexOffset];
                        const lastIndex = that.selectedIndex;
                        if (lastIndex !== index) {
                            if (lastIndex !== -1 && lastIndex === that.selectedCanvas.index)  redrawInputImage(that.selectedCanvas, lastIndex, false);
                            redrawInputImage(canvas, index, true);
                            that.selectedIndex = index;
                            that.selectedData = canvas.inputData;
                            that.selectedCanvas = canvas;
                        } else {
                            redrawInputImage(canvas, index, false);
                            that.selectedIndex = -1;
                            that.selectedData = null;
                            that.selectedCanvas = null;
                        }
                        that.redrawInnerLayer();
                        that.redrawOutputLayer();
                    }
                }
                while (container.children.length > countPerNumber)  container.removeChild(container.children[container.children.length-1]);
            }
            if (!parentContainer.children[11])  appendArrow(parentContainer, 1, countPerNumber> 1);

            redrawInputLayer(parentContainer);
        }

        function redrawInputLayer(parentContainer)
        {
            for (let j=0; j<10; j++)  {
                const container = parentContainer.children[j+1];
                for (let i=0; i<countPerNumber; i++)  {
                    const canvas = container.children[i];
                    const index = indicesByNumber[j][i + imageIndexOffset];
                    redrawInputImage(canvas, index, index === that.selectedIndex)
                }
            }
        }

        function redrawInputImage(canvas, index, selected)
        {
            if (index == null) {
                canvas.style.visibility = "hidden";
                return;
            }
            else  canvas.style.visibility = "visible";

            const x = inputData.slice([index,0], [1,IMAGE_SIZE]);
            canvas.inputData = x;
            canvas.index = index;
            const source = x.dataSync();
            if (source.length !== IMAGE_SIZE) throw "Wrong input vector size";

            const ctx = canvas.getContext('2d');
            const imgData = ctx.createImageData(IMAGE_W, IMAGE_H);
            const data = imgData.data;
            for (let i=0; i<data.length; i+=4) {
                let col = Math.round(source[i/4] * 255);
                data[i  ] = col;
                data[i+1] = col;
                data[i+2] = selected ? 255 : col;
                data[i+3] = 255;
            }
            ctx.putImageData(imgData, 0, 0);
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
                imageIndexOffset = mod(imageIndexOffset + direction * countPerNumber, maxNumberIndicesSize);
                redrawInputLayer(parent);
            };
            parent.appendChild(arrow);
        }

        window.addEventListener('resize', () => resize());
        resize();
    }

    initInnerLayer(containerId, n)
    {
        const fontMetricsBySize = [10,11,12,13,14].map((it) => getFontMetrics("monospace", it));

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

                    function createSpan(parent, text)  {
                        const span = document.createElement('span');
                        span.innerHTML = text;
                        parent.appendChild(span);
                        return span;
                    }
                    function createBr(parent)  {  parent.appendChild(document.createElement('br'));  }

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
                        el.numberRows = [];
                        for (let j=0; j<10; j++) {
                            const row = document.createElement('div');
                            el.appendChild(row);
                            el.numberRows.push(row);
                            row.numberLabel = createSpan(row, '');
                            createSpan(row, col1Sep);
                            row.weightLabel = createSpan(row, '');
                            createSpan(row, col2Sep);
                            row.outputLabel = createSpan(row, '');
                        }
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

        window.addEventListener('resize', () => resize());
        resize();
    }

    redrawInnerLayer() {
        let output = null;
        const inputData = this.selectedData;
        if (inputData) {
            const layerModel = tf.model({inputs: this.model.layers[0].input, outputs: this.model.layers[0].output});
            output = layerModel.predict(inputData).dataSync();
        }

        const containerId = 'innerLayer';
        const model = this.model;
        const container = document.getElementById(containerId);
        const weights = model.layers[0].trainableWeights[0];
        const biases = model.layers[0].trainableWeights[1];
        if (weights.shape[0] !== IMAGE_SIZE)  throw "Wrong layer size";
        const n = weights.shape[1];
        const weightsData = weights.val.dataSync();
        if (weightsData.length !== IMAGE_SIZE * n)  throw "Wrong layer weights data size";
        const biasesData = biases.val.dataSync();
        if (biasesData.length !== n)  throw "Wrong layer biases data size";
        const resultWeights = model.layers[1].trainableWeights[0];
        if (resultWeights.shape[0] !== n || resultWeights.shape[1] !== 10)  throw "Wrong output layer weights size";
        const resultWeightsData = resultWeights.val.dataSync();

        let maxOutput = -Infinity;
        if (output) {
            for (let j=0; j<n; j++) {
                for (let i=0; i<10; i++)  {
                    maxOutput = Math.max(maxOutput, output[j] * resultWeightsData[j * 10 + i]);
                }
            }
        }

        const weightPrecision = this.innerLayerWeightPrecision;
        function formatWeight(x)  {
            const a = Math.abs(x);
            return (x < 0 ? "−" : " ") + a.toFixed(weightPrecision - (a > 1 ? Math.ceil(Math.log10(a)) : 1));
        }

        for (let j=0; j<n; j++) {
            const div = container.items[j];
            div.biasLabel.innerHTML = "bias: &nbsp;&nbsp;" + formatWeight(biasesData[j]);
            div.outputLabel.innerHTML = "output:&nbsp;" + (output ? formatWeight(output[j]) : '');
            const k = j * 10;
            let outValues = Array.from(resultWeightsData.slice(k, k + 10))
                .map((v,i) => ({index: i, value: v}));
            if (this.sortOutputs)  outValues = outValues.sort((a,b) => Math.abs(b.value) - Math.abs(a.value));
            for (let i=0; i<10; i++)  {
                const item = outValues[i];
                const o = output ? output[j] * item.value : 0;
                div.numberRows[i].numberLabel.innerText = item.index.toString();
                div.numberRows[i].weightLabel.innerText = formatWeight(item.value);
                div.numberRows[i].outputLabel.innerText = output ? formatWeight(o) : '';
                const p = Math.abs(Math.round(o / maxOutput * 100));
                const col = o > 0 ? '#8f8' : '#f88';
                div.numberRows[i].outputLabel.style = output ? "background: linear-gradient(90deg, "+col+" "+p+"%, #fff "+p+"%)" : '';
            }
            this.redrawLayerWeights(div.canvas, weightsData, j, n);
        }
    }

    redrawLayerWeights(canvas, weightsData, j, n) {
        const inputData = this.selectedData && this.selectedData.dataSync();
        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(IMAGE_W, IMAGE_H);
        const data = imgData.data;
        for (let i=0; i<data.length; i+=4) {
            const w = weightsData[j + i/4 * n] * 4;
            let r,g,b;
            if (this.colored) {
                g = Math.round(Math.max(0,  w) * 255);
                r = Math.round(Math.max(0, -w) * 255);
                b = 0;
            }
            else  r = g = b = Math.round(clamp((w+1)/2, 0, 1) * 255);
            let sel = 0.4 * (inputData ? 1 - inputData[i/4] : 0);
            data[i  ] = Math.round(r * (1-sel) +   0 * sel);
            data[i+1] = Math.round(g * (1-sel) +   0 * sel);
            data[i+2] = Math.round(b * (1-sel) + 255 * sel);
            data[i+3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
    }

    initOutputLayer()
    {
        let container = document.getElementById('outputWeights');
        container.style.visibility = false;
        container.labels = [];
        container.bars = [];
        for (let i=0; i<10; i++)  {
            let label = document.createElement('span');
            label.innerText = i.toString();
            container.appendChild(label);
            container.labels.push(label);
            let bar = document.createElement('span');
            bar.className = 'bar';
            container.appendChild(bar);
            container.bars.push(bar);
        }
    }

    redrawOutputLayer()
    {
        let container = document.getElementById('outputWeights');
        container.style.visibility = this.selectedIndex !== -1;
        if (this.selectedIndex === -1)  return;

        let resultWeights = this.model.predict(this.selectedData).dataSync();
        let resultIndex = maxIndex(resultWeights, 0, resultWeights.length);
        let resultWeightsSum = resultWeights.reduce((a,b) => a+b, 0);

        for (let i=0; i<10; i++)  {
            container.labels[i].className = i === resultIndex ? 'winner' : '';
            container.bars[i].style.backgroundColor = resultWeights[i] >= 0 ? '#0a0' : '#a00';
            container.bars[i].style.height = Math.round(container.offsetHeight * Math.abs(resultWeights[i] / resultWeightsSum)) + 'px';
        }
    }

    setColorMode(colored) {
        this.colored = colored;
        this.redrawInnerLayer();  //todo only canvases
    }

    setSortOutputs(sortOutputs) {
        this.sortOutputs = sortOutputs;
        this.redrawInnerLayer();  //todo only labels
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