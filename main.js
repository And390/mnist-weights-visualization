
window.addEventListener('load', () => run());

async function run()
{
    const data = new MnistData();
    await data.load();

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

    initLayerContainer('hiddenLayer', HIDDEN_NODES);
    initLayerContainer('outLayer', 10);

    const trainData = data.getTrainData();
    const testData = data.getTestData();
    let xs = trainData.xs;
    xs = xs.reshape([xs.shape[0], IMAGE_SIZE]);
    const ys = trainData.labels;
    let testXS = testData.xs;
    testXS = testXS.reshape([testXS.shape[0], IMAGE_SIZE]);
    const testYS = trainData.labels;

    console.log('train...');

    let epoch = 0;
    model.fit(xs, ys, {epochs: 1000, callbacks: {onEpochEnd: () => {
        redrawLayerWeights(model.layers[0], 'hiddenLayer');
        console.log("epoch: " + (++epoch));
        if (epoch % 2 === 0)  {
            const n = testXS.shape[0];
            const result = model.predict(testXS).dataSync();
            const expected = testData.labels.dataSync();
            let success = 0;
            for (let i=0; i<n; i++) {
                const res = maxIndex(result, i*10, (i+1)*10);
                const exp = maxIndex(expected, i*10, (i+1)*10);
                if (res === exp)  success++;
            }
            console.log('result: ' + Math.round(success / n * 100) + '%');
        }
    }}});
}

function initLayerContainer(containerId, n)
{
    function resize()
    {
        const container = document.getElementById(containerId);
        const width = container.offsetWidth;
        const height = container.offsetHeight;
        const minSpace = 1;
        let s = 0;
        let numRows = 0;
        while (true)  {
            numRows++;
            const rowSize = Math.ceil(n / numRows);
            const scaleX = width / (IMAGE_W*rowSize + minSpace*(rowSize-1));
            const scaleY = height / (IMAGE_H*numRows + minSpace*(numRows-1));
            let scale = Math.min(scaleX, scaleY);
            scale = scale > 1 ? Math.floor(scale) : Math.pow(2, Math.floor(Math.log2(scale)));
            if (scale > s)  s = scale;  else break;
        }
        numRows--;
        let items = container.canvases;
        if (container.childElementCount !== numRows)  {
            if (items == null) {
                items = [];
                for (let i=0; i<n; i++) {
                    const el = document.createElement('canvas');
                    el.width = IMAGE_W;
                    el.height = IMAGE_H;
                    items.push(el);
                }
            }
            for (let r=0; r<container.children.length; r++)  {
                const row = container.children[r];
                for (let i=0; i<row.children.length; i++)  row.removeChild(row.children[i]);
            }
            while (container.children.length < numRows)  container.appendChild(document.createElement('div'));
            while (container.children.length > numRows)  container.removeChild(container.children[container.children.length-1]);
            const rowSize = Math.ceil(n / numRows);
            for (let r=0, j=0; r<numRows; r++)  for (let i=0; i<rowSize && j<items.length; i++, j++)  {
                container.children[r].appendChild(items[j]);
            }
            container.childElementCount = numRows;
            container.canvases = items;
        }
        for (let i=0; i<n; i++) {
            const el = items[i];
            el.style.width = Math.round(s * IMAGE_W) + 'px';
            el.style.height = Math.round(s * IMAGE_H) + 'px';
        }
    }

    window.addEventListener('resize', () => resize());
    resize();
}

function redrawLayerWeights(layer, containerId) {
    const container = document.getElementById(containerId);
    const weights = layer.trainableWeights[0];
    const biases = layer.trainableWeights[1];
    if (weights.shape[0] !== IMAGE_SIZE)  throw "Wrong layer size";
    const n = weights.shape[1];
    const weightsData = weights.val.dataSync();
    if (weightsData.length !== IMAGE_SIZE * n)  throw "Wrong layer weights data size";
    const biasesData = biases.val.dataSync();
    if (biasesData.length !== n)  throw "Wrong layer biases data size";
    for (let j=0; j<n; j++) {
        const canvas = container.canvases[j];
        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(IMAGE_W, IMAGE_H);
        const data = imgData.data;
        for (let i=0; i<data.length; i+=4) {
            const b = biasesData[j];
            const w = weightsData[j + i/4 * n] * 4;
            // let col = Math.round(clamp((Math.abs(w)) * 4, 0, 1) * 255);
            // data[i  ] = w < 0 ? col : 0;
            // data[i+1] = w > 0 ? col : 0;
            // data[i+2] = 0;
            // data[i+3] = 255;
            let col = Math.round(clamp((w+1)/2, 0, 1) * 255);
            data[i  ] = col;
            data[i+1] = col;
            data[i+2] = col;
            data[i+3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
    }
}

function drawInput(x)
{
    const source = x.dataSync();
    if (source.length !== IMAGE_SIZE) throw "Wrong input vector size";

    const canvas = document.getElementById("test_image");
    canvas.width = IMAGE_W;
    canvas.height = IMAGE_H;
    canvas.style.width = IMAGE_W*4 + "px";
    canvas.style.height = IMAGE_H*4 + "px";
    canvas.style.imageRendering = "pixelated";
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(IMAGE_W, IMAGE_H);
    const data = imgData.data;
    for (let i=0; i<data.length; i+=4) {
        let col = Math.round(source[i/4] * 255);
        data[i  ] = col;
        data[i+1] = col;
        data[i+2] = col;
        data[i+3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
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