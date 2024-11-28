const Hapi = require('@hapi/hapi');
const tf = require('@tensorflow/tfjs-node');
const Boom = require('@hapi/boom');
const uuid = require('uuid');
const { Storage } = require('@google-cloud/storage');
const { Firestore } = require('@google-cloud/firestore');
const path = require('path');
const fs = require('fs');
require("dotenv").config();



const init = async () => {
    const server = Hapi.server({
        port: parseInt(process.env.PORT),
        host: process.env.HOST,
        routes: {
            cors: {
              origin: ['https://asclepius-dot-submissionmlgc-alvinsetyap.et.r.appspot.com'],
            },
        },
    });


    const storage = new Storage({ keyFilename: "submissionmlgc-alvinsetyap-1687b533a136.json" });
    const firestore = new Firestore({ keyFilename: "submissionmlgc-alvinsetyap-1687b533a136.json" });

    const bucketName = process.env.BUCKET_NAME; 
    const modelPath = process.env.MODEL_DIR; 
    const localModelPath = path.join(__dirname, 'model');

    if (!fs.existsSync(localModelPath)) {
        fs.mkdirSync(localModelPath);
        await storage.bucket(bucketName).file(`${modelPath}/model.json`).download({ destination: path.join(localModelPath, 'model.json') });
        await storage.bucket(bucketName).file(`${modelPath}/group1-shard1of4.bin`).download({ destination: path.join(localModelPath, 'group1-shard1of4.bin') });
        await storage.bucket(bucketName).file(`${modelPath}/group1-shard2of4.bin`).download({ destination: path.join(localModelPath, 'group1-shard2of4.bin') });
        await storage.bucket(bucketName).file(`${modelPath}/group1-shard3of4.bin`).download({ destination: path.join(localModelPath, 'group1-shard3of4.bin') });
        await storage.bucket(bucketName).file(`${modelPath}/group1-shard4of4.bin`).download({ destination: path.join(localModelPath, 'group1-shard4of4.bin') });
    }

    const model = await tf.loadGraphModel(`file://${localModelPath}/model.json`);

    server.route({
        method: 'OPTIONS',
        path: '/{any*}',
        handler: (request, h) => {
            return h
                .response()
                .header('Access-Control-Allow-Origin', 'https://asclepius-dot-submissionmlgc-alvinsetyap.et.r.appspot.com')
                .header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
                .header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
        },
    });
    

    server.route({
        method: 'POST',
        path: '/predict',
        options: {
            payload: {
                output: 'stream',
                parse: true,
                multipart: true,
                maxBytes: 1000000,
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


                const imageTensor = tf.node
                    .decodeJpeg(buffer) 
                    .resizeNearestNeighbor([224, 224])
                    .expandDims()
                    .toFloat()

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
