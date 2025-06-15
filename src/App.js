import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as d3 from 'd3';
import * as mammoth from 'mammoth';

const BertopicStyleDocumentClustering = () => {
  const [documents, setDocuments] = useState([]);
  const [clustering, setClustering] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [pyodideStatus, setPyodideStatus] = useState({
    ready: false,
    loading: true,
    error: null
  });
  const svgRef = useRef(null);
  const fileInputRef = useRef(null);

  // Listen for global PDF processor status changes
  useEffect(() => {
    const handlePDFProcessorReady = (event) => {
      console.log('✅ Global PDF Processor ready!');
      setPyodideStatus({
        ready: true,
        loading: false,
        error: null
      });
    };

    const handlePDFProcessorError = (event) => {
      console.error('❌ Global PDF Processor error:', event.detail.error);
      setPyodideStatus({
        ready: false,
        loading: false,
        error: event.detail.error
      });
    };

    // Check current status
    if (window.getPyodideStatus) {
      const currentStatus = window.getPyodideStatus();
      setPyodideStatus(currentStatus);
    }

    // Listen for events
    window.addEventListener('globalPDFProcessorReady', handlePDFProcessorReady);
    window.addEventListener('globalPDFProcessorError', handlePDFProcessorError);

    // Poll for status updates during initialization
    const statusInterval = setInterval(() => {
      if (window.getPyodideStatus) {
        const currentStatus = window.getPyodideStatus();
        setPyodideStatus(currentStatus);
        
        // Stop polling once ready or error
        if (currentStatus.ready || currentStatus.error) {
          clearInterval(statusInterval);
        }
      }
    }, 1000);

    return () => {
      window.removeEventListener('globalPDFProcessorReady', handlePDFProcessorReady);
      window.removeEventListener('globalPDFProcessorError', handlePDFProcessorError);
      clearInterval(statusInterval);
    };
  }, []);

  // LIGHTWEIGHT TEXT SUMMARIZATION (fallback when Python not available)
  const lightweightSummarization = (text, maxSentences = 8) => {
    if (!text || text.length < 500) return text;

    // Split into sentences
    const sentences = text
      .replace(/([.!?])\s*(?=[A-Z])/g, "$1|")
      .split("|")
      .map(s => s.trim())
      .filter(s => s.length > 20 && s.length < 400);

    if (sentences.length <= maxSentences) return sentences.join(' ');

    // Enhanced scoring for abstract-style summaries
    const academicKeywords = ['research', 'analysis', 'study', 'method', 'result', 'conclusion', 'approach', 'findings', 'objective', 'demonstrate', 'significant', 'important'];
    
    const scoredSentences = sentences.map((sentence, index) => {
      let score = 0;
      
      // Position score (abstract structure)
      if (index === 0) score += 1.8; // First sentence often contains main point
      else if (index < sentences.length * 0.2) score += 1.5; // Early sentences - context
      else if (index > sentences.length * 0.8) score += 1.6; // Late sentences - conclusions
      else score += 1.0;
      
      // Academic keyword score
      const lowerSentence = sentence.toLowerCase();
      academicKeywords.forEach(keyword => {
        if (lowerSentence.includes(keyword)) score += 0.3;
      });
      
      // Length score (prefer substantial sentences for abstracts)
      const wordCount = sentence.split(' ').length;
      if (wordCount >= 15 && wordCount <= 35) score += 0.5;
      else if (wordCount > 35) score += 0.2;
      
      // Structure indicators
      if (lowerSentence.includes('this paper') || lowerSentence.includes('this study') || 
          lowerSentence.includes('we present') || lowerSentence.includes('our findings')) {
        score += 0.4;
      }
      
      return { sentence, score, index };
    });

    // Select top sentences and maintain order
    const topSentences = scoredSentences
      .sort((a, b) => b.score - a.score)
      .slice(0, maxSentences)
      .sort((a, b) => a.index - b.index);

    return topSentences.map(item => item.sentence).join(' ');
  };

  // LIGHTWEIGHT SENTENCE EMBEDDINGS (BERTopic-inspired without TensorFlow)
  const getLightweightSentenceEmbeddings = (sentences) => {
    // Create a vocabulary from all sentences
    const vocabulary = new Set();
    const processedSentences = sentences.map(sentence => {
      const words = sentence.toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(word => word.length > 3 && !stopWords.has(word));
      words.forEach(word => vocabulary.add(word));
      return words;
    });

    const vocabArray = Array.from(vocabulary).slice(0, 300); // Limit vocab size

    // Create TF-IDF-like embeddings with semantic enhancements
    const embeddings = processedSentences.map(words => {
      const vector = new Array(vocabArray.length).fill(0);
      const wordCount = words.length;
      
      // Term frequency with position weighting
      words.forEach((word, position) => {
        const index = vocabArray.indexOf(word);
        if (index !== -1) {
          // Position weighting (beginning and end of sentence matter more)
          const positionWeight = position === 0 || position === words.length - 1 ? 1.2 : 1.0;
          vector[index] += (1 / wordCount) * positionWeight;
        }
      });

      // Add semantic similarity via word co-occurrence patterns
      for (let i = 0; i < words.length - 1; i++) {
        const word1Index = vocabArray.indexOf(words[i]);
        const word2Index = vocabArray.indexOf(words[i + 1]);
        if (word1Index !== -1 && word2Index !== -1) {
          // Boost co-occurring terms
          vector[word1Index] += 0.1;
          vector[word2Index] += 0.1;
        }
      }
      
      // L2 normalization
      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      return magnitude > 0 ? vector.map(val => val / magnitude) : vector;
    });

    return embeddings;
  };

  // BERTOPIC-INSPIRED CLUSTERING PIPELINE (Enhanced with Python-style processing)
  const bertopicStyleClustering = async (documents) => {
    // Step 1: Extract and summarize text using Python (or fallback) - create abstract-style summaries
    const summaries = await Promise.all(
      documents.map(doc => pythonStyleSummarization(doc.content, 10))
    );
    
    // Step 2: Get lightweight sentence embeddings
    const embeddings = getLightweightSentenceEmbeddings(summaries);
    
    // Step 3: Dimensionality reduction (simple PCA-like)
    const reducedEmbeddings = reduceEmbeddingDimensions(embeddings);
    
    // Step 4: HDBSCAN-inspired clustering
    const clusters = performDensityClustering(reducedEmbeddings, documents.length);
    
    // Step 5: Build hierarchy with C-TF-IDF naming
    const hierarchy = buildBertopicHierarchy(documents, summaries, clusters, reducedEmbeddings);
    
    return hierarchy;
  };

  // SIMPLE DIMENSIONALITY REDUCTION
  const reduceEmbeddingDimensions = (embeddings) => {
    if (embeddings.length === 0) return [];
    
    // Simple approach: project to 2D using first two principal components
    const numDims = embeddings[0].length;
    const numPoints = embeddings.length;
    
    // Center the data
    const means = new Array(numDims).fill(0);
    embeddings.forEach(embedding => {
      embedding.forEach((val, i) => means[i] += val / numPoints);
    });
    
    const centeredData = embeddings.map(embedding => 
      embedding.map((val, i) => val - means[i])
    );
    
    // Simple projection to 2D (use first 2 dimensions or create them)
    return centeredData.map(point => [
      point.reduce((sum, val, i) => sum + val * Math.cos(i * 0.1), 0),
      point.reduce((sum, val, i) => sum + val * Math.sin(i * 0.1), 0)
    ]);
  };

  // BERTOPIC-STYLE DENSITY CLUSTERING
  const performDensityClustering = (embeddings, numDocs) => {
    // Dynamic parameters based on document count
    let minClusterSize, eps;
    if (numDocs < 20) {
      minClusterSize = 2;
      eps = 0.5;
    } else if (numDocs < 100) {
      minClusterSize = 3;
      eps = 0.4;
    } else {
      minClusterSize = Math.max(4, Math.floor(numDocs * 0.05));
      eps = 0.3;
    }

    const n = embeddings.length;
    const clusters = new Array(n).fill(-1);
    const visited = new Array(n).fill(false);
    let clusterId = 0;

    const euclideanDistance = (a, b) => {
      return Math.sqrt(a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0));
    };

    const getNeighbors = (pointIdx) => {
      const neighbors = [];
      for (let i = 0; i < n; i++) {
        if (i !== pointIdx && euclideanDistance(embeddings[pointIdx], embeddings[i]) <= eps) {
          neighbors.push(i);
        }
      }
      return neighbors;
    };

    // DBSCAN-style clustering
    for (let i = 0; i < n; i++) {
      if (visited[i]) continue;
      
      visited[i] = true;
      const neighbors = getNeighbors(i);
      
      if (neighbors.length < minClusterSize - 1) {
        continue; // Mark as noise
      }
      
      clusters[i] = clusterId;
      
      const queue = [...neighbors];
      while (queue.length > 0) {
        const neighborIdx = queue.shift();
        
        if (!visited[neighborIdx]) {
          visited[neighborIdx] = true;
          const neighborNeighbors = getNeighbors(neighborIdx);
          
          if (neighborNeighbors.length >= minClusterSize - 1) {
            queue.push(...neighborNeighbors.filter(idx => !queue.includes(idx)));
          }
        }
        
        if (clusters[neighborIdx] === -1) {
          clusters[neighborIdx] = clusterId;
        }
      }
      
      clusterId++;
    }

    return clusters;
  };

  // BUILD BERTOPIC-STYLE HIERARCHY WITH ENHANCED LABELING
  const buildBertopicHierarchy = (documents, summaries, clusters, embeddings) => {
    const clusterMap = new Map();
    const outliers = [];

    // Group documents by cluster and calculate confidence scores
    documents.forEach((doc, index) => {
      const clusterId = clusters[index];
      const confidence = calculateClusterConfidence(embeddings[index], embeddings, clusters, clusterId);
      
      const enhancedDoc = { 
        ...doc, 
        summary: summaries[index],
        embedding: embeddings[index],
        clusterId: clusterId,
        confidence: confidence,
        topicLabel: clusterId === -1 ? 'Outlier' : `Topic ${clusterId}`
      };

      if (clusterId === -1) {
        outliers.push({
          id: `outlier_${index}`,
          name: doc.name.replace(/\.[^/.]+$/, ""),
          size: doc.content.length,
          children: null,
          documents: [enhancedDoc],
          level: 1,
          topicInfo: {
            label: 'Outlier Document',
            keywords: extractDocumentKeywords(doc.content),
            confidence: 0.0,
            color: '#6b7280'
          }
        });
      } else {
        if (!clusterMap.has(clusterId)) {
          clusterMap.set(clusterId, []);
        }
        clusterMap.get(clusterId).push(enhancedDoc);
      }
    });

    const mainClusters = [];
    const topicColors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#84cc16', '#f97316'];

    // Create clusters with enhanced topic information
    Array.from(clusterMap.entries()).forEach(([clusterId, clusterDocs]) => {
      const topicInfo = generateEnhancedTopicInfo(clusterDocs, clusterId);
      const clusterName = topicInfo.label;
      const totalSize = clusterDocs.reduce((sum, doc) => sum + doc.content.length, 0);
      const color = topicColors[clusterId % topicColors.length];
      
      // Update document labels with topic info
      clusterDocs.forEach(doc => {
        doc.topicLabel = topicInfo.label;
        doc.topicKeywords = topicInfo.keywords;
        doc.topicColor = color;
      });
      
      // For larger clusters, create subclusters
      let children = null;
      if (clusterDocs.length > 8) {
        children = createSubClusters(clusterDocs, `cluster_${clusterId}`, topicInfo);
      } else {
        children = clusterDocs.map((doc, idx) => ({
          id: `cluster_${clusterId}_doc_${idx}`,
          name: doc.name.replace(/\.[^/.]+$/, ""),
          size: doc.content.length,
          children: null,
          documents: [doc],
          level: 2,
          topicInfo: topicInfo
        }));
      }

      mainClusters.push({
        id: `cluster_${clusterId}`,
        name: clusterName,
        size: totalSize,
        children: children,
        documents: clusterDocs,
        level: 1,
        topicInfo: {
          ...topicInfo,
          color: color,
          documentCount: clusterDocs.length,
          averageConfidence: clusterDocs.reduce((sum, doc) => sum + doc.confidence, 0) / clusterDocs.length
        }
      });
    });

    // Add outliers
    mainClusters.push(...outliers);

    if (mainClusters.length === 0) return null;
    if (mainClusters.length === 1) return mainClusters[0];

    return {
      id: 'root',
      name: 'Document Collection',
      size: mainClusters.reduce((sum, cluster) => sum + cluster.size, 0),
      children: mainClusters,
      documents: documents,
      level: 0,
      topicSummary: generateTopicSummary(mainClusters)
    };
  };

  // CALCULATE CLUSTER CONFIDENCE SCORE
  const calculateClusterConfidence = (docEmbedding, allEmbeddings, clusters, clusterId) => {
    if (clusterId === -1) return 0.0;
    
    // Find all documents in the same cluster
    const clusterEmbeddings = allEmbeddings.filter((_, index) => clusters[index] === clusterId);
    
    if (clusterEmbeddings.length <= 1) return 1.0;
    
    // Calculate average similarity to cluster members
    const similarities = clusterEmbeddings.map(embedding => {
      const dotProduct = docEmbedding.reduce((sum, val, i) => sum + val * embedding[i], 0);
      return Math.max(0, Math.min(1, dotProduct)); // Clamp between 0 and 1
    });
    
    const avgSimilarity = similarities.reduce((sum, sim) => sum + sim, 0) / similarities.length;
    return Math.round(avgSimilarity * 100) / 100; // Round to 2 decimal places
  };

  // GENERATE ENHANCED TOPIC INFORMATION
  const generateEnhancedTopicInfo = (clusterDocs, clusterId) => {
    const allText = clusterDocs.map(doc => doc.summary || doc.content).join(' ');
    const keywords = extractDocumentKeywords(allText);
    
    // Generate descriptive topic label
    const topKeywords = keywords.slice(0, 3).map(k => k.word);
    const label = topKeywords.length > 0 ? 
      `${topKeywords.join(', ')} Research` : 
      `Topic ${clusterId}`;
    
    return {
      id: clusterId,
      label: label,
      keywords: keywords,
      description: generateTopicDescription(keywords, clusterDocs.length),
      representativeTerms: topKeywords
    };
  };

  // EXTRACT DOCUMENT KEYWORDS
  const extractDocumentKeywords = (text) => {
    const words = text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 3 && !stopWords.has(word));

    // Count word frequencies
    const wordFreq = {};
    words.forEach(word => {
      wordFreq[word] = (wordFreq[word] || 0) + 1;
    });

    // Score and rank keywords
    const keywordScores = Object.entries(wordFreq).map(([word, freq]) => {
      let score = freq;
      
      // Boost academic terms
      if (['research', 'analysis', 'study', 'method', 'algorithm', 'learning', 'system', 'approach', 'model', 'framework', 'data', 'information', 'technology', 'science'].some(term => word.includes(term))) {
        score *= 2.0;
      }
      
      // Boost longer terms
      if (word.length > 6) score *= 1.3;
      
      return { word, score, frequency: freq };
    });

    return keywordScores
      .sort((a, b) => b.score - a.score)
      .slice(0, 8);
  };

  // GENERATE TOPIC DESCRIPTION
  const generateTopicDescription = (keywords, docCount) => {
    if (keywords.length === 0) return `Collection of ${docCount} documents`;
    
    const topKeywords = keywords.slice(0, 3).map(k => k.word);
    return `${docCount} documents about ${topKeywords.join(', ')}`;
  };

  // GENERATE TOPIC SUMMARY
  const generateTopicSummary = (clusters) => {
    const validClusters = clusters.filter(c => c.topicInfo && c.topicInfo.id !== undefined);
    
    return {
      totalTopics: validClusters.length,
      totalDocuments: clusters.reduce((sum, c) => sum + (c.documents ? c.documents.length : 0), 0),
      topTopics: validClusters
        .sort((a, b) => b.documents.length - a.documents.length)
        .slice(0, 5)
        .map(cluster => ({
          label: cluster.topicInfo.label,
          documentCount: cluster.documents.length,
          keywords: cluster.topicInfo.representativeTerms,
          confidence: cluster.topicInfo.averageConfidence
        }))
    };
  };

  const generateBertopicClusterName = (clusterDocs) => {
    if (clusterDocs.length === 0) return "Empty Cluster";
    if (clusterDocs.length === 1) {
      return clusterDocs[0].name.replace(/\.[^/.]+$/, "");
    }

    // Extract key terms from summaries
    const allText = clusterDocs.map(doc => doc.summary || doc.content).join(' ');
    const words = allText.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 3 && !stopWords.has(word));

    // Count term frequencies
    const termFreq = {};
    words.forEach(word => {
      termFreq[word] = (termFreq[word] || 0) + 1;
    });

    // Score terms using simplified C-TF-IDF
    const termScores = Object.entries(termFreq).map(([term, freq]) => {
      let score = freq / clusterDocs.length; // TF
      
      // Boost academic/technical terms
      if (['research', 'analysis', 'study', 'method', 'algorithm', 'learning', 'system', 'approach', 'model', 'framework'].some(keyword => term.includes(keyword))) {
        score *= 2.0;
      }
      
      // Boost longer terms
      if (term.length > 6) score *= 1.3;
      
      return { term, score };
    });

    // Get top terms
    const topTerms = termScores
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map(item => item.term.charAt(0).toUpperCase() + item.term.slice(1));

    if (topTerms.length === 0) return "Mixed Documents";
    if (topTerms.length === 1) return `${topTerms[0]} Research`;
    if (topTerms.length === 2) return `${topTerms[0]} & ${topTerms[1]}`;
    return `${topTerms[0]}, ${topTerms[1]} & ${topTerms[2]}`;
  };

  // CREATE SUBCLUSTERS for large clusters
  const createSubClusters = (clusterDocs, parentId, parentTopicInfo) => {
    const midpoint = Math.floor(clusterDocs.length / 2);
    
    const subCluster1Docs = clusterDocs.slice(0, midpoint);
    const subCluster2Docs = clusterDocs.slice(midpoint);
    
    return [
      {
        id: `${parentId}_sub_1`,
        name: generateBertopicClusterName(subCluster1Docs),
        size: subCluster1Docs.reduce((sum, doc) => sum + doc.content.length, 0),
        children: subCluster1Docs.map((doc, idx) => ({
          id: `${parentId}_sub_1_doc_${idx}`,
          name: doc.name.replace(/\.[^/.]+$/, ""),
          size: doc.content.length,
          children: null,
          documents: [doc],
          level: 3
        })),
        documents: subCluster1Docs,
        level: 2,
        topicInfo: {
          ...parentTopicInfo,
          label: `${parentTopicInfo.label} (Part A)`,
          documentCount: subCluster1Docs.length
        }
      },
      {
        id: `${parentId}_sub_2`,
        name: generateBertopicClusterName(subCluster2Docs),
        size: subCluster2Docs.reduce((sum, doc) => sum + doc.content.length, 0),
        children: subCluster2Docs.map((doc, idx) => ({
          id: `${parentId}_sub_2_doc_${idx}`,
          name: doc.name.replace(/\.[^/.]+$/, ""),
          size: doc.content.length,
          children: null,
          documents: [doc],
          level: 3
        })),
        documents: subCluster2Docs,
        level: 2,
        topicInfo: {
          ...parentTopicInfo,
          label: `${parentTopicInfo.label} (Part B)`,
          documentCount: subCluster2Docs.length
        }
      }
    ];
  };

  // STOP WORDS
  const stopWords = new Set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
    'his', 'its', 'our', 'their', 'from', 'up', 'out', 'down', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'also', 'time', 'way'
  ]);

  // FILE PARSING FUNCTIONS
  const parseTextFile = (file) => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve({ title: '', content: e.target.result });
      reader.readAsText(file);
    });
  };

  const parseWordDoc = async (file) => {
    try {
      const arrayBuffer = await file.arrayBuffer();
      const result = await mammoth.extractRawText({ arrayBuffer });
      return { title: '', content: result.value };
    } catch (error) {
      console.error('Error parsing Word document:', error);
      return { title: '', content: '' };
    }
  };

  // ENHANCED FILE PARSING with Global PDF Processor
  const parseFile = async (file) => {
    const extension = file.name.split('.').pop().toLowerCase();
    
    console.log(`📄 Processing file: ${file.name} (${extension})`);
    console.log('🔍 PDF Processor Status:', {
      ready: pyodideStatus.ready,
      loading: pyodideStatus.loading,
      error: pyodideStatus.error,
      pdfSupport: pyodideStatus.pdfSupport
    });
    
    // Use global PDF processor for PDF files if available
    if (extension === 'pdf') {
      console.log('📄 PDF file detected, checking processor availability...');
      
      if (!pyodideStatus.ready) {
        console.warn('⚠️ PDF Processor not ready, using fallback');
        return await parseFallback(file, extension);
      }
      
      if (!window.processFileWithPython) {
        console.warn('⚠️ processFileWithPython function not available, using fallback');
        return await parseFallback(file, extension);
      }
      
      try {
        console.log('🐍 Processing PDF with Global Python Processor...');
        const arrayBuffer = await file.arrayBuffer();
        console.log(`📊 PDF file size: ${arrayBuffer.byteLength} bytes`);
        
        const result = await window.processFileWithPython(arrayBuffer);
        console.log('🔍 PDF processing result:', result);
        
        if (!result.success) {
          console.warn('⚠️ PDF processing failed, using fallback:', result.error);
          return await parseFallback(file, extension);
        }
        
        console.log('✅ PDF processed successfully:', {
          charCount: result.char_count,
          wordCount: result.word_count,
          pageCount: result.page_count,
          hasKeywords: result.keywords.length > 0,
          processingMethod: result.processing_method
        });
        
        return { 
          title: '', 
          content: result.text,
          summary: result.summary,
          keywords: result.keywords,
          metadata: {
            charCount: result.char_count,
            wordCount: result.word_count,
            pageCount: result.page_count,
            processingMethod: result.processing_method
          }
        };
        
      } catch (error) {
        console.error('❌ Error processing PDF with Global Python:', error);
        console.error('Error details:', error.message, error.stack);
        // Fallback to basic parsing
        return await parseFallback(file, extension);
      }
    } else {
      // Use regular parsing for non-PDF files
      console.log(`📝 Processing ${extension} file with standard parser`);
      return await parseFallback(file, extension);
    }
  };

  const parseFallback = async (file, extension) => {
    try {
      if (extension === 'txt') {
        return await parseTextFile(file);
      } else if (extension === 'docx' || extension === 'doc') {
        return await parseWordDoc(file);
      } else if (extension === 'pdf') {
        // Enhanced PDF fallback with better error reporting
        console.warn(`⚠️ PDF processing unavailable for ${file.name}`);
        console.log('Attempting basic PDF text extraction...');
        
        try {
          // Try to read as text (might work for some simple PDFs)
          const text = await file.text();
          if (text && text.length > 100 && !text.includes('%PDF')) {
            console.log('✅ Basic text extraction worked');
            return { 
              title: '', 
              content: text.replace(/[^\x20-\x7E\n]/g, ' '),
              metadata: {
                charCount: text.length,
                wordCount: text.split(/\s+/).length,
                processingMethod: 'basic_text'
              }
            };
          }
        } catch (e) {
          console.warn('Basic text extraction failed:', e.message);
        }
        
        // Enhanced fallback message with debugging info
        const fallbackContent = `PDF Document: ${file.name}

⚠️ PROCESSING STATUS:
- File Size: ${(file.size / 1024).toFixed(1)} KB
- Global PDF Processor: ${pyodideStatus.ready ? 'Ready' : 'Not Ready'}
- PDF Support: ${pyodideStatus.pdfSupport ? 'Available' : 'Unavailable'}
- Error: Content extraction failed

POSSIBLE CAUSES:
• PDF is password-protected
• PDF contains only images (scanned document)
• PDF uses unsupported encoding
• PDF is corrupted or invalid
• File is too large to process

SOLUTIONS:
• Try converting PDF to text first
• Use OCR software for image-based PDFs
• Check if PDF opens normally in other applications
• Try uploading a different PDF to test the system

This document will be processed as text-only for clustering analysis.`;

        return { 
          title: '', 
          content: fallbackContent,
          metadata: {
            charCount: fallbackContent.length,
            wordCount: fallbackContent.split(/\s+/).length,
            processingMethod: 'pdf_fallback',
            originalFileName: file.name,
            originalFileSize: file.size
          }
        };
      } else {
        // For other files, try basic text extraction
        const text = await file.text();
        return { title: '', content: text.replace(/[^\x20-\x7E]/g, ' ') };
      }
    } catch (error) {
      console.error('Fallback parsing failed:', error);
      return { 
        title: '', 
        content: `Error processing file: ${file.name}\nError: ${error.message}`,
        metadata: {
          charCount: 50,
          wordCount: 10,
          processingMethod: 'error_fallback'
        }
      };
    }
  };

  // PYTHON-STYLE TEXT SUMMARIZATION using global functions
  const pythonStyleSummarization = async (text, maxSentences = 10) => {
    // Use Global Python processor if available
    if (pyodideStatus.ready && text && text.length > 500) {
      try {
        console.log('🐍 Using Global Python summarization for abstract...');
        const summary = await window.summarizeTextWithPython(text, maxSentences);
        return summary;
      } catch (error) {
        console.warn('Python summarization failed, using fallback:', error);
      }
    }
    
    // Fallback to lightweight summarization
    return lightweightSummarization(text, maxSentences);
  };

  // ENHANCED SUNBURST VISUALIZATION
  const createBertopicSunburst = useCallback((data) => {
    if (!data || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 900;
    const height = 900;
    const radius = Math.min(width, height) / 2 - 70;

    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#84cc16', '#f97316'];
    const getColor = (d) => {
      if (d.depth === 0) return '#f8f9fa';
      const rootParent = d.ancestors().find(ancestor => ancestor.depth === 1);
      const colorIndex = rootParent ? rootParent.data.id.charCodeAt(rootParent.data.id.length - 1) % colors.length : 0;
      const baseColor = d3.color(colors[colorIndex]);
      return baseColor.brighter(d.depth * 0.2).toString();
    };

    const partition = d3.partition().size([2 * Math.PI, radius]);
    const arc = d3.arc()
      .startAngle(d => d.x0)
      .endAngle(d => d.x1)
      .innerRadius(d => d.y0)
      .outerRadius(d => d.y1);

    svg.attr("width", width).attr("height", height);
    const g = svg.append("g").attr("transform", `translate(${width / 2},${height / 2})`);

    const root = d3.hierarchy(data).sum(d => d.children ? 0 : d.size);
    partition(root);

    const paths = g.selectAll("path")
      .data(root.descendants())
      .enter().append("path")
      .attr("d", arc)
      .style("fill", getColor)
      .style("stroke", "#fff")
      .style("stroke-width", 2)
      .style("opacity", d => d.depth === 0 ? 0 : 0.8)
      .style("cursor", d => d.depth > 0 ? "pointer" : "default");

    paths
      .on("mouseover", function(event, d) {
        if (d.depth === 0) return;
        d3.select(this).style("opacity", 1);
        
        const tooltip = g.append("g").attr("class", "tooltip");
        tooltip.append("rect")
          .attr("x", -150).attr("y", -60)
          .attr("width", 300).attr("height", 120)
          .attr("rx", 10)
          .style("fill", "rgba(0,0,0,0.9)")
          .style("stroke", "#3b82f6")
          .style("stroke-width", 2);
        
        const text = tooltip.append("text")
          .attr("text-anchor", "middle")
          .style("fill", "white");
        
        text.append("tspan").attr("x", 0).attr("dy", -25)
          .style("font-size", "14px").style("font-weight", "bold")
          .text(d.data.name);
        
        if (d.data.documents) {
          text.append("tspan").attr("x", 0).attr("dy", 20)
            .style("font-size", "12px")
            .text(`${d.data.documents.length} documents • Level ${d.depth}`);
          
          // Show topic information if available
          if (d.data.topicInfo) {
            text.append("tspan").attr("x", 0).attr("dy", 16)
              .style("font-size", "11px").style("fill", "#94a3b8")
              .text(`Topic: ${d.data.topicInfo.label}`);
            
            if (d.data.topicInfo.representativeTerms) {
              text.append("tspan").attr("x", 0).attr("dy", 14)
                .style("font-size", "10px").style("fill", "#64748b")
                .text(`Keywords: ${d.data.topicInfo.representativeTerms.slice(0, 3).join(', ')}`);
            }
            
            if (d.data.topicInfo.averageConfidence !== undefined) {
              text.append("tspan").attr("x", 0).attr("dy", 14)
                .style("font-size", "10px").style("fill", "#64748b")
                .text(`Avg Confidence: ${(d.data.topicInfo.averageConfidence * 100).toFixed(1)}%`);
            }
          } else {
            text.append("tspan").attr("x", 0).attr("dy", 16)
              .style("font-size", "11px").style("fill", "#94a3b8")
              .text(`Size: ${d.value.toLocaleString()} chars`);
          }
        }
      })
      .on("mouseout", function(event, d) {
        if (d.depth === 0) return;
        d3.select(this).style("opacity", 0.8);
        g.selectAll(".tooltip").remove();
      })
      .on("click", function(event, d) {
        if (d.depth === 0 || !d.data.documents) return;
        event.stopPropagation();
        setSelectedCluster({
          name: d.data.name,
          documents: d.data.documents,
          level: d.data.level || d.depth,
          subClusters: d.children ? d.children.length : 0
        });
      });

    // Center label
    const centerGroup = g.append("g");
    centerGroup.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "-1em")
      .style("font-size", "24px")
      .style("font-weight", "bold")
      .style("fill", "#1f2937")
      .text("🐍 Global PDF Processor");
    
    centerGroup.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "1em")
      .style("font-size", "16px")
      .style("fill", "#6b7280")
      .text(`${documents.length} Documents`);

  }, [documents]);

  // MAIN PROCESSING FUNCTION
  const processFiles = async () => {
    if (selectedFiles.length === 0) return;

    setIsProcessing(true);
    const processedDocs = [];
    const processingResults = {
      successful: [],
      failed: [],
      fallback: []
    };

    console.log(`🚀 Starting to process ${selectedFiles.length} files...`);

    for (const file of selectedFiles) {
      try {
        console.log(`\n📄 Processing: ${file.name}`);
        const sections = await parseFile(file);

        const content = [sections.title, sections.content].filter(Boolean).join(' ');

        if (content.length < 50) {
          console.warn(`⚠️ Skipping ${file.name}: insufficient content (${content.length} chars)`);
          processingResults.failed.push({
            fileName: file.name,
            reason: 'Insufficient content',
            contentLength: content.length
          });
          continue;
        }

        // Track processing results
        if (sections.metadata?.processingMethod === 'python_pypdf2') {
          processingResults.successful.push({
            fileName: file.name,
            method: 'Python PDF Processing',
            charCount: sections.metadata.charCount,
            pageCount: sections.metadata.pageCount
          });
        } else if (sections.metadata?.processingMethod === 'pdf_fallback') {
          processingResults.fallback.push({
            fileName: file.name,
            reason: 'PDF extraction failed',
            originalSize: sections.metadata.originalFileSize
          });
        }

        processedDocs.push({
          name: file.name,
          title: sections.title,
          content: content,
          size: file.size,
          summary: sections.summary,
          keywords: sections.keywords,
          metadata: sections.metadata
        });

        console.log(`✅ Processed ${file.name}: ${content.length} characters`);
      } catch (error) {
        console.error(`❌ Error processing ${file.name}:`, error);
        processingResults.failed.push({
          fileName: file.name,
          reason: error.message,
          error: error
        });
      }
    }

    // Log processing summary
    console.log('\n📊 PROCESSING SUMMARY:');
    console.log(`✅ Successful PDF processing: ${processingResults.successful.length}`);
    console.log(`⚠️ PDF fallback used: ${processingResults.fallback.length}`);
    console.log(`❌ Failed processing: ${processingResults.failed.length}`);
    
    if (processingResults.fallback.length > 0) {
      console.log('\n⚠️ PDFs that used fallback:');
      processingResults.fallback.forEach(item => {
        console.log(`  • ${item.fileName} (${(item.originalSize / 1024).toFixed(1)} KB) - ${item.reason}`);
      });
    }

    if (processingResults.failed.length > 0) {
      console.log('\n❌ Failed files:');
      processingResults.failed.forEach(item => {
        console.log(`  • ${item.fileName} - ${item.reason}`);
      });
    }

    setDocuments(processedDocs);
    
    if (processedDocs.length > 0) {
      try {
        console.log(`\n🔄 Starting clustering analysis for ${processedDocs.length} documents...`);
        const hierarchyResult = await bertopicStyleClustering(processedDocs);
        setClustering(hierarchyResult);
        console.log('✅ Clustering completed successfully');
      } catch (error) {
        console.error('❌ Error in clustering:', error);
        alert('Clustering failed. Please try with fewer or smaller documents.');
      }
    }
    
    setIsProcessing(false);

    // Show processing summary to user if there were issues
    if (processingResults.fallback.length > 0 || processingResults.failed.length > 0) {
      const message = `Processing completed with some issues:

✅ Successfully processed: ${processingResults.successful.length} files
⚠️ Used fallback processing: ${processingResults.fallback.length} files  
❌ Failed to process: ${processingResults.failed.length} files

${processingResults.fallback.length > 0 ? '\nPDFs using fallback:\n' + processingResults.fallback.map(f => `• ${f.fileName}`).join('\n') : ''}

Check browser console for detailed information.`;
      
      alert(message);
    }
  };

  useEffect(() => {
    if (clustering) {
      createBertopicSunburst(clustering);
    }
  }, [clustering, createBertopicSunburst]);

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
  };

  const removeFile = (index) => {
    setSelectedFiles(selectedFiles.filter((_, i) => i !== index));
  };

  const getPyodideStatusInfo = () => {
    if (pyodideStatus.loading) {
      return {
        icon: '🔄',
        text: 'Loading Global PDF Processor...',
        color: '#f59e0b',
        bgColor: '#fef3c7'
      };
    } else if (pyodideStatus.ready) {
      const hasProcessFunction = typeof window.processFileWithPython === 'function';
      const hasSummarizeFunction = typeof window.summarizeTextWithPython === 'function';
      
      if (hasProcessFunction && hasSummarizeFunction) {
        return {
          icon: '🐍',
          text: `Global PDF Processor Ready (${pyodideStatus.pdfSupport ? 'Full PDF Support' : 'Text Only'})`,
          color: '#16a34a',
          bgColor: '#dcfce7'
        };
      } else {
        return {
          icon: '⚠️',
          text: 'PDF Processor Partially Ready (Missing Functions)',
          color: '#f59e0b',
          bgColor: '#fef3c7'
        };
      }
    } else if (pyodideStatus.error) {
      return {
        icon: '⚠️',
        text: `PDF Processor Failed: ${pyodideStatus.error}`,
        color: '#dc2626',
        bgColor: '#fee2e2'
      };
    } else {
      return {
        icon: '❓',
        text: 'PDF Processor Status Unknown',
        color: '#6b7280',
        bgColor: '#f3f4f6'
      };
    }
  };

  // TEST FUNCTION - Add this for debugging
  const testPDFProcessor = async () => {
    console.log('🧪 Testing PDF Processor...');
    console.log('Status:', window.getPyodideStatus());
    console.log('processFileWithPython:', typeof window.processFileWithPython);
    console.log('summarizeTextWithPython:', typeof window.summarizeTextWithPython);
    
    if (window.summarizeTextWithPython) {
      try {
        const testText = `This research presents a comprehensive analysis of document clustering techniques in modern information retrieval systems. The study investigates the effectiveness of various clustering algorithms when applied to large-scale document collections. Our methodology employs both traditional approaches and novel machine learning techniques to evaluate clustering performance. The experimental setup includes datasets from multiple domains including academic papers, news articles, and technical documentation. Results demonstrate that hybrid clustering approaches significantly outperform single-algorithm methods, achieving up to 15% improvement in clustering accuracy. The findings suggest that combining semantic similarity measures with traditional tf-idf vectorization provides optimal clustering results. Furthermore, the research reveals important insights about scalability challenges in real-world applications. These results have significant implications for information retrieval systems and recommend specific implementation strategies for practical deployment.`;
        
        const testSummary = await window.summarizeTextWithPython(testText, 10);
        console.log('✅ Test abstract-style summarization result:');
        console.log(testSummary);
        console.log(`📊 Original: ${testText.length} chars, Summary: ${testSummary.length} chars`);
      } catch (error) {
        console.error('❌ Test summarization failed:', error);
      }
    }
  };

  const statusInfo = getPyodideStatusInfo();

  return (
    <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      <div style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1 style={{ fontSize: '2.5rem', fontWeight: '800', color: '#1f2937', marginBottom: '10px' }}>
          🐍 Global PDF Processor + Document Clustering
        </h1>
        <p style={{ fontSize: '1.1rem', color: '#6b7280', marginBottom: '8px' }}>
          Global PDF Processing + Lightweight Text Analysis + BERTopic-Inspired Clustering
        </p>
        <div style={{ 
          display: 'inline-flex', 
          alignItems: 'center', 
          gap: '8px',
          padding: '8px 16px',
          backgroundColor: statusInfo.bgColor,
          borderRadius: '20px',
          fontSize: '0.9rem',
          fontWeight: '500'
        }}>
          <span style={{ 
            width: '8px', 
            height: '8px', 
            borderRadius: '50%', 
            backgroundColor: statusInfo.color
          }}></span>
          {statusInfo.icon} {statusInfo.text}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', marginBottom: '30px' }}>
        {/* File Upload Section */}
        <div style={{ background: 'white', padding: '25px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#1f2937', marginBottom: '20px' }}>
            📁 Upload Documents
          </h2>
          
          <input
            type="file"
            ref={fileInputRef}
            multiple
            accept=".txt,.pdf,.doc,.docx"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
          
          <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
            <button
              onClick={() => fileInputRef.current?.click()}
              style={{
                padding: '12px 20px',
                backgroundColor: '#3b82f6',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: '500'
              }}
            >
              📂 Select Files
            </button>
            <button
              onClick={processFiles}
              disabled={selectedFiles.length === 0 || isProcessing}
              style={{
                padding: '12px 20px',
                backgroundColor: selectedFiles.length === 0 || isProcessing ? '#9ca3af' : pyodideStatus.ready ? '#10b981' : '#f59e0b',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: selectedFiles.length === 0 || isProcessing ? 'not-allowed' : 'pointer',
                fontWeight: '500'
              }}
            >
              {isProcessing ? '🐍 Processing...' : pyodideStatus.ready ? '🚀 Python Analysis' : '⚡ Basic Analysis'}
            </button>
            {pyodideStatus.ready && (
              <button
                onClick={testPDFProcessor}
                style={{
                  padding: '12px 20px',
                  backgroundColor: '#6366f1',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontWeight: '500'
                }}
              >
                🧪 Test PDF Processor
              </button>
            )}
          </div>

          {selectedFiles.length > 0 && (
            <div style={{ marginBottom: '20px' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: '500', marginBottom: '10px' }}>
                Selected Files ({selectedFiles.length}):
              </h3>
              <div style={{ maxHeight: '200px', overflowY: 'auto', border: '1px solid #e5e7eb', borderRadius: '6px' }}>
                {selectedFiles.map((file, index) => (
                  <div key={index} style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    padding: '10px',
                    borderBottom: index < selectedFiles.length - 1 ? '1px solid #f3f4f6' : 'none'
                  }}>
                    <div>
                      <div style={{ fontWeight: '500' }}>{file.name}</div>
                      <div style={{ fontSize: '0.8rem', color: '#6b7280' }}>
                        {(file.size / 1024).toFixed(1)} KB
                        {file.name.toLowerCase().endsWith('.pdf') && (
                          <span style={{ 
                            marginLeft: '8px',
                            padding: '2px 6px',
                            backgroundColor: pyodideStatus.ready ? '#dcfce7' : '#fee2e2',
                            color: pyodideStatus.ready ? '#166534' : '#dc2626',
                            borderRadius: '8px',
                            fontSize: '0.7rem'
                          }}>
                            {pyodideStatus.ready ? '🐍 PDF Ready' : '⚠️ Basic Only'}
                          </span>
                        )}
                      </div>
                    </div>
                    <button
                      onClick={() => removeFile(index)}
                      style={{
                        width: '24px',
                        height: '24px',
                        backgroundColor: '#ef4444',
                        color: 'white',
                        border: 'none',
                        borderRadius: '50%',
                        cursor: 'pointer'
                      }}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {documents.length > 0 && (
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: '500', marginBottom: '10px' }}>
                Processed Documents ({documents.length}):
              </h3>
              <div style={{ maxHeight: '200px', overflowY: 'auto', border: '1px solid #e5e7eb', borderRadius: '6px' }}>
                {documents.map((doc, index) => (
                  <div key={index} style={{ 
                    padding: '10px',
                    borderBottom: index < documents.length - 1 ? '1px solid #f3f4f6' : 'none'
                  }}>
                    <div style={{ fontWeight: '500' }}>{doc.name}</div>
                    <div style={{ fontSize: '0.8rem', color: '#6b7280', display: 'flex', gap: '15px', flexWrap: 'wrap' }}>
                      <span>{doc.content.length} characters</span>
                      {doc.metadata && (
                        <>
                          <span>{doc.metadata.wordCount} words</span>
                          {doc.metadata.pageCount && <span>{doc.metadata.pageCount} pages</span>}
                          {doc.summary && <span>📝 Summary</span>}
                          {doc.keywords && doc.keywords.length > 0 && <span>🔑 {doc.keywords.length} keywords</span>}
                          <span style={{ 
                            padding: '1px 6px',
                            backgroundColor: doc.metadata.processingMethod === 'python_pypdf2' ? '#dcfce7' : '#fef3c7',
                            color: doc.metadata.processingMethod === 'python_pypdf2' ? '#166534' : '#92400e',
                            borderRadius: '8px',
                            fontSize: '0.7rem'
                          }}>
                            {doc.metadata.processingMethod === 'python_pypdf2' ? '🐍 Python' : '⚡ Basic'}
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Visualization Section */}
        <div style={{ background: 'white', padding: '25px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#1f2937', marginBottom: '20px' }}>
            🎯 Topic Clusters
          </h2>
          
          {clustering ? (
            <div style={{ textAlign: 'center' }}>
              <svg ref={svgRef}></svg>
            </div>
          ) : (
            <div style={{ 
              textAlign: 'center', 
              padding: '80px 20px',
              border: '2px dashed #d1d5db',
              borderRadius: '8px',
              color: '#6b7280'
            }}>
              <div style={{ fontSize: '4rem', marginBottom: '20px' }}>🐍</div>
              <p style={{ fontSize: '1.1rem', fontWeight: '500', marginBottom: '8px' }}>
                Upload documents for Global PDF processing
              </p>
              <p style={{ fontSize: '0.9rem', marginBottom: '6px' }}>
                🐍 PyPDF2 for reliable PDF text extraction
              </p>
              <p style={{ fontSize: '0.9rem', marginBottom: '6px' }}>
                📝 Python-powered text summarization
              </p>
              <p style={{ fontSize: '0.9rem', marginBottom: '6px' }}>
                🎯 BERTopic-inspired clustering pipeline
              </p>
              <p style={{ fontSize: '0.9rem' }}>
                🏷️ C-TF-IDF automatic topic labeling
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Selected Cluster Details */}
      {selectedCluster && (
        <div style={{ 
          background: 'white', 
          padding: '25px', 
          borderRadius: '12px', 
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
          marginBottom: '30px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px' }}>
            <div style={{ flex: 1 }}>
              <h3 style={{ fontSize: '1.6rem', fontWeight: '700', color: '#1f2937', marginBottom: '8px' }}>
                📁 {selectedCluster.name}
              </h3>
              {selectedCluster.documents[0]?.topicLabel && (
                <div style={{ 
                  display: 'inline-block',
                  padding: '6px 12px',
                  backgroundColor: selectedCluster.documents[0]?.topicColor || '#e5e7eb',
                  color: 'white',
                  borderRadius: '16px',
                  fontSize: '0.9rem',
                  fontWeight: '600',
                  marginBottom: '8px'
                }}>
                  🏷️ {selectedCluster.documents[0].topicLabel}
                </div>
              )}
              {selectedCluster.documents[0]?.topicKeywords && (
                <div style={{ fontSize: '0.9rem', color: '#6b7280' }}>
                  <strong>Key Terms:</strong> {selectedCluster.documents[0].topicKeywords.slice(0, 5).map(k => k.word).join(', ')}
                </div>
              )}
            </div>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexShrink: 0 }}>
              <span style={{
                padding: '6px 14px',
                backgroundColor: '#dbeafe',
                color: '#1e40af',
                borderRadius: '20px',
                fontSize: '0.875rem',
                fontWeight: '600'
              }}>
                Level {selectedCluster.level} • {selectedCluster.documents.length} doc{selectedCluster.documents.length > 1 ? 's' : ''}
              </span>
              <button
                onClick={() => setSelectedCluster(null)}
                style={{
                  width: '32px',
                  height: '32px',
                  backgroundColor: '#ef4444',
                  color: 'white',
                  border: 'none',
                  borderRadius: '50%',
                  cursor: 'pointer',
                  fontSize: '16px'
                }}
              >
                ✕
              </button>
            </div>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '20px' }}>
            {selectedCluster.documents.map((doc, index) => (
              <div key={index} style={{ 
                border: '2px solid',
                borderColor: doc.topicColor || '#e5e7eb',
                borderRadius: '12px', 
                padding: '18px',
                backgroundColor: '#f9fafb',
                position: 'relative'
              }}>
                {/* Document Label Badge */}
                {doc.topicLabel && (
                  <div style={{
                    position: 'absolute',
                    top: '-10px',
                    left: '15px',
                    padding: '4px 12px',
                    backgroundColor: doc.topicColor || '#6b7280',
                    color: 'white',
                    borderRadius: '12px',
                    fontSize: '0.75rem',
                    fontWeight: '600'
                  }}>
                    {doc.topicLabel}
                  </div>
                )}
                
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '12px', marginTop: '8px' }}>
                  <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#1f2937', margin: 0 }}>
                    📄 {doc.name}
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'end', gap: '4px' }}>
                    <div style={{ fontSize: '0.8rem', color: '#6b7280' }}>
                      {(doc.size / 1024).toFixed(1)} KB
                    </div>
                    {doc.confidence !== undefined && (
                      <div style={{
                        padding: '2px 8px',
                        backgroundColor: doc.confidence > 0.7 ? '#dcfce7' : doc.confidence > 0.4 ? '#fef3c7' : '#fee2e2',
                        color: doc.confidence > 0.7 ? '#166534' : doc.confidence > 0.4 ? '#92400e' : '#dc2626',
                        borderRadius: '10px',
                        fontSize: '0.75rem',
                        fontWeight: '600'
                      }}>
                        {(doc.confidence * 100).toFixed(0)}% confidence
                      </div>
                    )}
                    {doc.metadata && (
                      <div style={{
                        padding: '2px 8px',
                        backgroundColor: doc.metadata.processingMethod === 'python_pypdf2' ? '#dcfce7' : '#fef3c7',
                        color: doc.metadata.processingMethod === 'python_pypdf2' ? '#166534' : '#92400e',
                        borderRadius: '10px',
                        fontSize: '0.75rem',
                        fontWeight: '600'
                      }}>
                        {doc.metadata.processingMethod === 'python_pypdf2' ? '🐍 Global Python' : '⚡ Basic'}
                      </div>
                    )}
                  </div>
                </div>
                
                {doc.summary && (
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ fontSize: '0.875rem', fontWeight: '500', color: '#374151', marginBottom: '6px' }}>
                      📝 {doc.metadata?.processingMethod === 'python_pypdf2' ? 'Python' : 'Basic'} Summary:
                    </div>
                    <div style={{ fontSize: '0.9rem', color: '#4b5563', lineHeight: '1.5', fontStyle: 'italic' }}>
                      {doc.summary.length > 200 ? doc.summary.substring(0, 200) + "..." : doc.summary}
                    </div>
                  </div>
                )}
                
                {/* Document Keywords */}
                {doc.topicKeywords && (
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ fontSize: '0.875rem', fontWeight: '500', color: '#374151', marginBottom: '6px' }}>
                      🔑 Topic Keywords:
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                      {doc.topicKeywords.slice(0, 6).map((keyword, idx) => (
                        <span key={idx} style={{
                          padding: '4px 10px',
                          backgroundColor: doc.topicColor || '#e5e7eb',
                          color: 'white',
                          borderRadius: '15px',
                          fontSize: '0.75rem',
                          fontWeight: '500'
                        }}>
                          {keyword.word} ({keyword.frequency})
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                
                <div style={{ fontSize: '0.8rem', color: '#6b7280', borderTop: '1px solid #e5e7eb', paddingTop: '8px' }}>
                  Cluster ID: {doc.clusterId !== undefined ? doc.clusterId : 'N/A'} • {doc.content.length} characters
                  {doc.metadata && (
                    <>
                      <span> • {doc.metadata.wordCount} words</span>
                      {doc.metadata.pageCount && <span> • {doc.metadata.pageCount} pages</span>}
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
          
          {/* Topic Summary */}
          {selectedCluster.documents.length > 1 && selectedCluster.documents[0]?.topicKeywords && (
            <div style={{ 
              marginTop: '25px', 
              padding: '20px', 
              backgroundColor: '#f0f9ff', 
              borderRadius: '12px',
              border: '2px solid #bae6fd'
            }}>
              <h4 style={{ fontSize: '1.1rem', fontWeight: '700', color: '#0c4a6e', marginBottom: '12px' }}>
                📊 Topic Analysis Summary
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
                <div>
                  <div style={{ fontSize: '0.9rem', fontWeight: '600', color: '#374151' }}>Documents:</div>
                  <div style={{ fontSize: '1.1rem', color: '#0c4a6e', fontWeight: '700' }}>{selectedCluster.documents.length}</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.9rem', fontWeight: '600', color: '#374151' }}>Avg Confidence:</div>
                  <div style={{ fontSize: '1.1rem', color: '#0c4a6e', fontWeight: '700' }}>
                    {selectedCluster.documents.filter(d => d.confidence !== undefined).length > 0 ? 
                      `${(selectedCluster.documents.reduce((sum, doc) => sum + (doc.confidence || 0), 0) / selectedCluster.documents.length * 100).toFixed(1)}%` : 
                      'N/A'
                    }
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.9rem', fontWeight: '600', color: '#374151' }}>Total Size:</div>
                  <div style={{ fontSize: '1.1rem', color: '#0c4a6e', fontWeight: '700' }}>
                    {(selectedCluster.documents.reduce((sum, doc) => sum + doc.size, 0) / 1024).toFixed(1)} KB
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.9rem', fontWeight: '600', color: '#374151' }}>Processing:</div>
                  <div style={{ fontSize: '1rem', color: '#0c4a6e', fontWeight: '600' }}>
                    {pyodideStatus.ready ? '🐍 Global Python' : '⚡ Basic Analysis'}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Algorithm Explanation */}
      <div style={{ 
        background: 'white', 
        padding: '25px', 
        borderRadius: '12px', 
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
      }}>
        <h3 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#1f2937', marginBottom: '20px' }}>
          🧠 Global PDF Processing Pipeline
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
          <div>
            <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#1f2937', marginBottom: '8px' }}>
              🌐 Global Setup:
            </h4>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: '#4b5563' }}>
              <li style={{ marginBottom: '4px' }}>• Loads once in index.html</li>
              <li style={{ marginBottom: '4px' }}>• PyPDF2 for reliable PDF extraction</li>
              <li style={{ marginBottom: '4px' }}>• Accessible from React components</li>
              <li style={{ marginBottom: '4px' }}>• Event-driven status management</li>
            </ul>
          </div>
          <div>
            <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#1f2937', marginBottom: '8px' }}>
              📄 PDF Processing:
            </h4>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: '#4b5563' }}>
              <li style={{ marginBottom: '4px' }}>• Page-by-page text extraction</li>
              <li style={{ marginBottom: '4px' }}>• Clean text normalization</li>
              <li style={{ marginBottom: '4px' }}>• Keyword extraction & scoring</li>
              <li style={{ marginBottom: '4px' }}>• Extractive summarization</li>
            </ul>
          </div>
          <div>
            <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#1f2937', marginBottom: '8px' }}>
              🎯 Clustering:
            </h4>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: '#4b5563' }}>
              <li style={{ marginBottom: '4px' }}>• Lightweight embeddings</li>
              <li style={{ marginBottom: '4px' }}>• DBSCAN-inspired clustering</li>
              <li style={{ marginBottom: '4px' }}>• C-TF-IDF topic labeling</li>
              <li style={{ marginBottom: '4px' }}>• Interactive visualization</li>
            </ul>
          </div>
        </div>
        
        <div style={{ 
          marginTop: '20px', 
          padding: '16px', 
          backgroundColor: '#f0f9ff', 
          borderRadius: '8px',
          border: '1px solid #bae6fd'
        }}>
          <p style={{ margin: 0, color: '#0c4a6e', fontWeight: '500' }}>
            🚀 <strong>Global & Reliable:</strong> This implementation uses a global Pyodide instance 
            loaded once in index.html, making PDF processing available to all React components. 
            Perfect for document analysis applications requiring consistent PDF text extraction!
          </p>
        </div>
      </div>
    </div>
  );
};

export default BertopicStyleDocumentClustering;