const Hapi = require('@hapi/hapi');
const tf = require('@tensorflow/tfjs-node');
const Boom = require('@hapi/boom');
const uuid = require('uuid');
const { Storage } = require('@google-cloud/storage');
const { Firestore } = require('@google-cloud/firestore');
const path = require('path');
const fs = require('fs');


const keyFilePath = "submissionmlgc-alvinsetyap-1687b533a136.json"

const init = async () => {
    const server = Hapi.server({
        port: 8000,
        host: '127.0.0.1',
        routes: {
            cors: {
              origin: ['*'],
            },
        },
    });

    // Initialize Google Cloud Storage and Firestore
    const storage = new Storage({ keyFilename: keyFilePath });
    const firestore = new Firestore({ keyFilename: keyFilePath });

    // Load the model from the cloud bucket
    const bucketName = 'submissionmlgc-alvinsetyap.appspot.com'; // Replace with your bucket name
    const modelPath = 'submissions-model'; // Replace with the path to your model directory in the bucket
    const localModelPath = path.join(__dirname, 'model');

    if (!fs.existsSync(localModelPath)) {
        fs.mkdirSync(localModelPath);
        console.log('Downloading model from bucket...');
        await storage.bucket(bucketName).file(`${modelPath}/model.json`).download({ destination: path.join(localModelPath, 'model.json') });
        await storage.bucket(bucketName).file(`${modelPath}/group1-shard1of4.bin`).download({ destination: path.join(localModelPath, 'group1-shard1of4.bin') });
        await storage.bucket(bucketName).file(`${modelPath}/group1-shard2of4.bin`).download({ destination: path.join(localModelPath, 'group1-shard2of4.bin') });
        await storage.bucket(bucketName).file(`${modelPath}/group1-shard3of4.bin`).download({ destination: path.join(localModelPath, 'group1-shard3of4.bin') });
        await storage.bucket(bucketName).file(`${modelPath}/group1-shard4of4.bin`).download({ destination: path.join(localModelPath, 'group1-shard4of4.bin') });
    }

    const model = await tf.loadGraphModel(`file://${localModelPath}/model.json`);

    server.route({
        method: 'POST',
        path: '/predict',
        options: {
            payload: {
                output: 'stream',
                parse: true,
                multipart: true,
                maxBytes: 1000000, // Set maximum allowed payload size
            },
        },
        handler: async (request, h) => {
            try {
                const { payload } = request;
                

                if (!payload.image) {
                    return h.response({
                        status: "fail",
                        message: "Image is required",
                        data : 0
                     }).code(400)
                }

                const file = payload.image;
                const chunks = [];

                for await (const chunk of file) {
                    chunks.push(chunk);
                }

                const buffer = Buffer.concat(chunks);

                if (buffer.length > 1000000) {
                    return h.response({
                        status: "fail",
                        message: "Payload content length greater than maximum allowed: 1000000",
                        data : 0
                     }).code(413)
                }

                // Preprocess the image directly from the buffer
                const imageTensor = tf.node
                    .decodeJpeg(buffer) // Decode image as RGB
                    .resizeNearestNeighbor([224, 224]) // Resize to model input shape (224x224)
                    .expandDims()
                    .toFloat()

                // Perform prediction
                const prediction = model.predict(imageTensor);
                // const predictedClass = tf.argMax(prediction, 1).dataSync()[0];
                const score = await prediction.data()

                const confidientScore = Math.max(...score) * 100
                let result = 0
                
                if (confidientScore > 0.5) {
                    result = 1
                } else  {
                    result = 0            
                }

                // Map prediction to response
                const resultMap = {
                    0: {
                        result: 'Non-cancer',
                        suggestion: 'Penyakit kanker tidak terdeteksi.',
                    },
                    1: {
                        result: 'Cancer',
                        suggestion: 'Segera periksa ke dokter!',
                    },
                };

                const response = resultMap[result] || {
                    result: 'Unknown',
                    suggestion: 'Hasil tidak dapat diinterpretasikan.',
                };

                // Create prediction result
                const predictionResult = {
                    id: uuid.v4(),
                    result: response.result,
                    suggestion: response.suggestion,
                    createdAt: new Date().toISOString(),
                };


                // Store result in Firestore
                await firestore.collection('predictions').doc(predictionResult.id).set(predictionResult);

                return h.response({
                    status: 'success',
                    message: 'Model is predicted successfully',
                    data: predictionResult,
                }).code(201);
            } catch (error) {
                if (Boom.isBoom(error, 413)) {
                    return h.response({
                        status: "fail",
                        message: "Payload content length greater than maximum allowed: 1000000",
                        data : 0
                     }).code(413)
                }

                return h.response({
                    status: "fail",
                    message: "Terjadi kesalahan dalam melakukan prediksi",
                    data : 0
                 }).code(400)
            }
        },
    });

    await server.start();
    console.log(`Server running on ${server.info.uri}`);
};

process.on('unhandledRejection', (err) => {
    console.error(err);
    process.exit(1);
});

init();
