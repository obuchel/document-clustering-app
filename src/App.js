import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as d3 from 'd3';
import * as mammoth from 'mammoth';

const EnhancedDocumentClusteringApp = () => {
  const [documents, setDocuments] = useState([]);
  const [clustering, setClustering] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [selectedCluster, setSelectedCluster] = useState(null);
  const svgRef = useRef(null);
  const fileInputRef = useRef(null);

  // Enhanced preprocessing with aggressive metadata removal (from your functions2.py)
  const metadataTerms = [
    "'", '.', ',', '"', "'", '&', "'", "'s", '(', ')', '+', ',,', '-', '-.', '.', 
    '.-.', '/', ':', '|', '[', ']', '*', '.02', '.04', 'sub', '<', '>', '/sub', '/i', 
    '%', '=', '2018', '2019', '2019.', '2020', '2020-2021', ':', ';', '?', '``', 
    'nlmcategory=', 'abstracttext=', 'abstracttext', 'label=', 'label', '1', '2', '95', 
    '4', '3', '6', '5', '60', '20', '10', '2023', '2022', '8', '333', '584', '972', 
    '1140', '0', '1', '12', '30', '40', '25', '2011', '2016', '80', '2017', '100', 
    '50', '15', '2015', "''", "{", "}", '2021', 'obj', 'endobj', 'stream', 'endstream',
    
    // ALL PDF METADATA TERMS FROM YOUR SAMPLE DATA:
    'filter', 'encoding', 'winansiencoding', 'baseencoding', 'macromanencoding',
    'description', 'pagenumber', 'lastmodified', 'version', 'linearized', 'openaction',
    'dest', 'border', 'subsection', 'caption', 'producer', 'conformance', 'iccbased',
    'indexed', 'trailer', 'application', 'shading', 'function', 'luminosity', 'infinity',
    'reference', 'placement', 'sourceautocomputed', 'converted', 'registration',
    'courier', 'identity', 'uity', 'process', 'action', 'leading', 'representation',
    'gts_pdfxversion', 'header', 'sequence', 'progression', 'visualization',
    
    // ENCODING COMBINATIONS FROM YOUR SAMPLE:
    'encoding winansiencoding', 'encoding macromanencoding', 'iccbased filter',
    
    // PDF NOISE PATTERNS FROM YOUR SAMPLE:
    'qxed', 'xoer', 'deed', 'taed', 'ejer', 'aoed', 'bied', 'zved', 'qser', 'daer',
    'pjued', 'boed', 'nfed', 'wger', 'zubaker', 'qinserted', 'gts', 'tter', 'exer',
    'nter', 'pted', 'trapped', 'ged', 'ueed', 'kled', 'mer', 'ped',
    
    // DOCUMENT STRUCTURE TERMS:
    'document', 'section', 'chapter', 'part', 'figure', 'table', 'content', 'text', 'page', 'paper', 'pdf'
  ];



  const preprocessText = (text) => {
    // Create regex pattern from metadata terms
    const metadataPattern = new RegExp('\\b(?:' + metadataTerms.map(term => 
      term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    ).join('|') + ')\\b', 'gi');

    let cleanedText = text
      .toLowerCase()
      // Remove metadata terms first
      .replace(metadataPattern, ' ')
      // Extra aggressive PDF filtering for the main culprits
      .replace(/\bfilter\b/g, ' ')
      .replace(/\bencoding\b/g, ' ')
      .replace(/\bwinansiencoding\b/g, ' ')
      .replace(/\bdescription\b/g, ' ')
      .replace(/\bpagenumber\b/g, ' ')
      .replace(/\blastmodified\b/g, ' ')
      .replace(/\bversion\b/g, ' ')
      .replace(/\bborder\b/g, ' ')
      .replace(/\bproducer\b/g, ' ')
      .replace(/\bindexed\b/g, ' ')
      .replace(/\bshading\b/g, ' ')
      .replace(/\bfunction\b/g, ' ')
      .replace(/\bsequence\b/g, ' ')
      // Remove all numeric sequences
      .replace(/\b\d+\b/g, ' ')
      .replace(/\b0{2,}\b/g, ' ')
      .replace(/\b[0-9a-f]{6,}\b/g, ' ')
      // Remove alphanumeric codes
      .replace(/\b[a-z0-9]{1,3}\b/g, ' ')
      .replace(/\b[a-z]+\d+\b/g, ' ')
      .replace(/\b\d+[a-z]+\b/g, ' ')
      // Remove special characters and normalize
      .replace(/[^\w\s]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

    // Additional cleaning pass
    cleanedText = cleanedText
      .split(' ')
      .filter(word => {
        if (word.length < 2) return false;
        if (word.length === 2 && !/^(is|it|to|of|in|on|at|by|or|an|as|be|do|go|he|me|my|no|so|up|we|if)$/.test(word)) return false;
        if (word.length === 3 && !/^(the|and|for|are|but|not|you|all|can|had|her|was|one|our|out|day|get|has|him|his|how|man|new|now|old|see|two|way|who)$/.test(word)) return false;
        
        // Extra filter for PDF metadata that might slip through
        if (['filter', 'encoding', 'winansiencoding', 'description', 'pagenumber', 'lastmodified', 
             'version', 'border', 'producer', 'indexed', 'shading', 'function', 'sequence'].includes(word)) {
          return false;
        }
        
        return true;
      })
      .join(' ')
      .trim();

    return cleanedText;
  };

  // Enhanced stop words (from your functions2.py)
  const stopWords = new Set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
    'his', 'its', 'our', 'their', 'from', 'up', 'out', 'down', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'also', 'one',
    'patients', 'health', 'disease', 'study', 'research', 'analysis', 'results', 'data',
    'document', 'section', 'chapter', 'part', 'figure', 'table', 'content', 'text', 'page'
  ]);

  // Extract meaningful terms with POS-like filtering
  const extractTerms = (text) => {
    const cleanText = preprocessText(text);
    
    if (cleanText.length < 50) {
      return {};
    }

    const words = cleanText.split(/\s+/).filter(word => {
      if (word.length < 4) return false;
      if (word.length > 30) return false;
      if (stopWords.has(word)) return false;
      if (!/[a-zA-Z]/.test(word)) return false;
      if (!/[aeiou]/.test(word)) return false;
      if (/(.)\1{2,}/.test(word)) return false;
      return true;
    });

    const terms = {};

    // Extract unigrams
    words.forEach(word => {
      if (isHighValueTerm(word)) {
        terms[word] = (terms[word] || 0) + 1;
      }
    });

    // Extract bigrams
    for (let i = 0; i < words.length - 1; i++) {
      if (isHighValueTerm(words[i]) && isHighValueTerm(words[i + 1])) {
        const bigram = `${words[i]} ${words[i + 1]}`;
        if (isMeaningfulPhrase(bigram)) {
          terms[bigram] = (terms[bigram] || 0) + 1;
        }
      }
    }

    return terms;
  };

  const isHighValueTerm = (term) => {
    const meaningfulPatterns = [
      /.*tion$/, /.*sion$/, /.*ment$/, /.*ness$/, /.*ity$/, /.*ence$/, /.*ance$/,
      /.*ing$/, /.*ed$/, /.*er$/, /.*est$/, /.*ful$/, /.*less$/, /.*able$/, /.*ible$/
    ];
    
    const meaningfulWords = new Set([
      'research', 'analysis', 'method', 'approach', 'development', 'management', 
      'information', 'technology', 'science', 'education', 'business', 'process',
      'design', 'model', 'framework', 'strategy', 'implementation', 'evaluation'
    ]);

    return term.length >= 4 && 
           (meaningfulPatterns.some(pattern => pattern.test(term)) || 
            meaningfulWords.has(term)) &&
           /[aeiou]/.test(term) &&
           !/[0-9]/.test(term);
  };

  const isMeaningfulPhrase = (phrase) => {
    const words = phrase.split(' ');
    return words.length === 2 && 
           words.every(word => isHighValueTerm(word)) &&
           words[0] !== words[1];
  };

  // File parsing functions
  const parseTextFile = (file) => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target.result;
        resolve({ title: '', content: text });
      };
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

  const parsePDF = async (file) => {
    try {
      const text = await file.text();
      return { title: '', content: text.replace(/[^\x20-\x7E]/g, ' ') };
    } catch (error) {
      console.error('Error parsing PDF:', error);
      return { title: '', content: '' };
    }
  };

  // TF-IDF embeddings
  const createTFIDFEmbeddings = (documents) => {
    const termCounts = {};
    const docCount = documents.length;

    documents.forEach(doc => {
      const uniqueTerms = new Set(Object.keys(doc.terms));
      uniqueTerms.forEach(term => {
        termCounts[term] = (termCounts[term] || 0) + 1;
      });
    });

    const vocabulary = Object.entries(termCounts)
      .filter(([term, count]) => count >= Math.max(1, Math.floor(docCount * 0.05)) && count <= Math.floor(docCount * 0.8))
      .sort(([,a], [,b]) => b - a)
      .slice(0, 500)
      .map(([term]) => term);

    return documents.map(doc => {
      const vector = new Array(vocabulary.length).fill(0);
      const totalTerms = Object.values(doc.terms).reduce((sum, count) => sum + count, 0);

      vocabulary.forEach((term, index) => {
        const tf = (doc.terms[term] || 0) / totalTerms;
        const idf = Math.log(docCount / (termCounts[term] || 1));
        vector[index] = tf * idf;
      });

      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      return magnitude > 0 ? vector.map(val => val / magnitude) : vector;
    });
  };

  // Enhanced HDBSCAN-like clustering with hierarchy (inspired by your hdb.py)
  const hierarchicalClustering = (embeddings, documents) => {
    const n = embeddings.length;
    
    // Parameters based on document count (from your hdb.py)
    let minClusterSize;
    if (n < 50) {
      minClusterSize = 2;
    } else if (n >= 50 && n < 1000) {
      minClusterSize = 4;
    } else {
      minClusterSize = 20;
    }

    // Level 1 clustering (broad topics)
    const level1Clusters = dbscanClustering(embeddings, 0.3, minClusterSize);
    
    // Create hierarchy with subdivision for large clusters
    const hierarchy = buildHierarchicalStructure(documents, embeddings, level1Clusters, minClusterSize);
    
    return hierarchy;
  };

  const dbscanClustering = (embeddings, eps, minPts) => {
    const n = embeddings.length;
    const visited = new Array(n).fill(false);
    const clusters = new Array(n).fill(-1);
    let clusterId = 0;

    const cosineSimilarity = (a, b) => {
      const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
      const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
      const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
      return magnitudeA && magnitudeB ? dotProduct / (magnitudeA * magnitudeB) : 0;
    };

    const getNeighbors = (pointIdx) => {
      const neighbors = [];
      for (let i = 0; i < n; i++) {
        if (i !== pointIdx && cosineSimilarity(embeddings[pointIdx], embeddings[i]) >= eps) {
          neighbors.push(i);
        }
      }
      return neighbors;
    };

    for (let i = 0; i < n; i++) {
      if (visited[i]) continue;
      
      visited[i] = true;
      const neighbors = getNeighbors(i);
      
      if (neighbors.length < minPts - 1) {
        continue;
      }
      
      clusters[i] = clusterId;
      
      for (let j = 0; j < neighbors.length; j++) {
        const neighborIdx = neighbors[j];
        
        if (!visited[neighborIdx]) {
          visited[neighborIdx] = true;
          const neighborNeighbors = getNeighbors(neighborIdx);
          
          if (neighborNeighbors.length >= minPts - 1) {
            neighbors.push(...neighborNeighbors.filter(idx => !neighbors.includes(idx)));
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

  // Build hierarchical structure with subdivision (like your prepare_data.py)
// Build hierarchical structure with deep subdivision
// Build hierarchical structure with deep subdivision
const buildHierarchicalStructure = (documents, embeddings, level1Clusters, minClusterSize) => {
  const clusterMap = new Map();
  const noiseDocs = [];

  // Group documents by level 1 cluster
  documents.forEach((doc, index) => {
    const clusterId = level1Clusters[index];
    if (clusterId === -1) {
      noiseDocs.push({ doc, embedding: embeddings[index], index });
    } else {
      if (!clusterMap.has(clusterId)) {
        clusterMap.set(clusterId, []);
      }
      clusterMap.get(clusterId).push({ doc, embedding: embeddings[index], index });
    }
  });

  const clusters = [];

  // Process each level 1 cluster with deep recursion
  Array.from(clusterMap.entries()).forEach(([clusterId, clusterData]) => {
    const clusterDocs = clusterData.map(item => item.doc);
    const clusterEmbeddings = clusterData.map(item => item.embedding);
    
    // Use recursive subdivision for deeper hierarchies
    const deepSubClusters = recursiveSubdivision(
      clusterDocs, 
      clusterEmbeddings, 
      minClusterSize, 
      1, // Starting at level 1
      `cluster_${clusterId}`,
      documents
    );
    
    clusters.push(deepSubClusters);
  });

  // Add noise documents as individual clusters
  noiseDocs.forEach((item, index) => {
    clusters.push({
      id: `outlier_${index}`,
      name: item.doc.name.replace(/\.[^/.]+$/, ""),
      size: Object.values(item.doc.terms).reduce((a, b) => a + b, 0),
      children: null,
      documents: [item.doc],
      level: 1
    });
  });

  if (clusters.length === 0) return null;
  if (clusters.length === 1) return clusters[0];

  return {
    id: 'root',
    name: 'Document Collection',
    size: clusters.reduce((sum, cluster) => sum + cluster.size, 0),
    children: clusters,
    documents: documents,
    level: 0
  };
};

// Recursive subdivision for deeper hierarchies (up to 6 levels)
const recursiveSubdivision = (clusterDocs, clusterEmbeddings, minClusterSize, currentLevel, parentId, allDocs) => {
  const maxDepth = 6; // Allow up to 6 levels deep
  const minDocsForSubdivision = Math.max(3, minClusterSize); // Minimum docs needed for subdivision
  
  // Base case: too few documents, too deep, or subdivision threshold not met
  if (clusterDocs.length < minDocsForSubdivision || 
      currentLevel >= maxDepth || 
      clusterDocs.length < (currentLevel === 1 ? 6 : 4)) {
    
    // Create leaf nodes for individual documents
    if (clusterDocs.length <= 3) {
      return {
        id: parentId,
        name: generateClusterName(clusterDocs, allDocs),
        size: clusterDocs.reduce((sum, doc) => sum + Object.values(doc.terms).reduce((a, b) => a + b, 0), 0),
        children: clusterDocs.map((doc, idx) => ({
          id: `${parentId}_doc_${idx}`,
          name: doc.name.replace(/\.[^/.]+$/, ""),
          size: Object.values(doc.terms).reduce((a, b) => a + b, 0),
          children: null,
          documents: [doc],
          level: currentLevel + 1
        })),
        documents: clusterDocs,
        level: currentLevel
      };
    } else {
      // Medium-sized cluster - create one more level
      return {
        id: parentId,
        name: generateClusterName(clusterDocs, allDocs),
        size: clusterDocs.reduce((sum, doc) => sum + Object.values(doc.terms).reduce((a, b) => a + b, 0), 0),
        children: clusterDocs.map((doc, idx) => ({
          id: `${parentId}_doc_${idx}`,
          name: doc.name.replace(/\.[^/.]+$/, ""),
          size: Object.values(doc.terms).reduce((a, b) => a + b, 0),
          children: null,
          documents: [doc],
          level: currentLevel + 1
        })),
        documents: clusterDocs,
        level: currentLevel
      };
    }
  }

  // Calculate epsilon based on current level (tighter clustering at deeper levels)
  const epsilon = Math.max(0.2, 0.6 - (currentLevel * 0.1));
  const minPts = Math.max(2, Math.floor(minClusterSize / Math.pow(2, currentLevel - 1)));
  
  // Perform clustering at current level
  const subClusters = dbscanClustering(clusterEmbeddings, epsilon, minPts);
  const subClusterMap = new Map();
  const subNoise = [];

  // Group documents by sub-cluster
  clusterDocs.forEach((doc, index) => {
    const subClusterId = subClusters[index];
    if (subClusterId === -1) {
      subNoise.push({ doc, embedding: clusterEmbeddings[index] });
    } else {
      if (!subClusterMap.has(subClusterId)) {
        subClusterMap.set(subClusterId, []);
      }
      subClusterMap.get(subClusterId).push({ 
        doc, 
        embedding: clusterEmbeddings[index] 
      });
    }
  });

  const children = [];

  // Recursively process each sub-cluster
  Array.from(subClusterMap.entries()).forEach(([subClusterId, subClusterData]) => {
    const subClusterDocs = subClusterData.map(item => item.doc);
    const subClusterEmbeddings = subClusterData.map(item => item.embedding);
    
    const childCluster = recursiveSubdivision(
      subClusterDocs,
      subClusterEmbeddings,
      minClusterSize,
      currentLevel + 1,
      `${parentId}_sub_${subClusterId}`,
      allDocs
    );
    
    children.push(childCluster);
  });

  // Add noise documents as individual nodes
  subNoise.forEach((item, index) => {
    children.push({
      id: `${parentId}_noise_${index}`,
      name: item.doc.name.replace(/\.[^/.]+$/, ""),
      size: Object.values(item.doc.terms).reduce((a, b) => a + b, 0),
      children: null,
      documents: [item.doc],
      level: currentLevel + 1
    });
  });

  // If no meaningful subdivision occurred, create document leaves
  if (children.length <= 1 && clusterDocs.length > 1) {
    return {
      id: parentId,
      name: generateClusterName(clusterDocs, allDocs),
      size: clusterDocs.reduce((sum, doc) => sum + Object.values(doc.terms).reduce((a, b) => a + b, 0), 0),
      children: clusterDocs.map((doc, idx) => ({
        id: `${parentId}_doc_${idx}`,
        name: doc.name.replace(/\.[^/.]+$/, ""),
        size: Object.values(doc.terms).reduce((a, b) => a + b, 0),
        children: null,
        documents: [doc],
        level: currentLevel + 1
      })),
      documents: clusterDocs,
      level: currentLevel
    };
  }

  return {
    id: parentId,
    name: generateClusterName(clusterDocs, allDocs),
    size: clusterDocs.reduce((sum, doc) => sum + Object.values(doc.terms).reduce((a, b) => a + b, 0), 0),
    children: children,
    documents: clusterDocs,
    level: currentLevel
  };
};

// Recursive subdivision for deeper hierarchies (up to 6 levels)


  // Subdivide large clusters (inspired by your multi-level approach)
  /*const subdivideCluster = (clusterDocs, clusterEmbeddings, minClusterSize) => {
    // Apply secondary clustering with tighter parameters
    const subClusters = dbscanClustering(clusterEmbeddings, 0.5, Math.max(2, Math.floor(minClusterSize / 2)));
    
    const subClusterMap = new Map();
    const subNoise = [];

    clusterDocs.forEach((doc, index) => {
      const subClusterId = subClusters[index];
      if (subClusterId === -1) {
        subNoise.push(doc);
      } else {
        if (!subClusterMap.has(subClusterId)) {
          subClusterMap.set(subClusterId, []);
        }
        subClusterMap.get(subClusterId).push(doc);
      }
    });

    const result = [];

    // Create subclusters
    Array.from(subClusterMap.entries()).forEach(([subClusterId, subClusterDocs]) => {
      result.push({
        id: `subcluster_${subClusterId}`,
        name: generateClusterName(subClusterDocs, clusterDocs),
        size: subClusterDocs.reduce((sum, doc) => sum + Object.values(doc.terms).reduce((a, b) => a + b, 0), 0),
        children: subClusterDocs.map((doc, idx) => ({
          id: `subdoc_${subClusterId}_${idx}`,
          name: doc.name.replace(/\.[^/.]+$/, ""),
          size: Object.values(doc.terms).reduce((a, b) => a + b, 0),
          children: null,
          documents: [doc],
          level: 3
        })),
        documents: subClusterDocs,
        level: 2
      });
    });

    // Add noise documents
    subNoise.forEach((doc, index) => {
      result.push({
        id: `subnoise_${index}`,
        name: doc.name.replace(/\.[^/.]+$/, ""),
        size: Object.values(doc.terms).reduce((a, b) => a + b, 0),
        children: null,
        documents: [doc],
        level: 2
      });
    });

    return result;
  };*/

  // Enhanced cluster naming with c-TF-IDF (from your approach)
  const generateClusterName = (clusterDocs, allDocs) => {
    if (clusterDocs.length === 0) return "Empty Cluster";
    if (clusterDocs.length === 1) {
      const doc = clusterDocs[0];
      return doc.name.replace(/\.[^/.]+$/, "").length > 30 ? 
        doc.name.replace(/\.[^/.]+$/, "").substring(0, 27) + "..." : 
        doc.name.replace(/\.[^/.]+$/, "");
    }

    // c-TF-IDF approach
    const clusterTermFreq = {};
    const clusterSize = clusterDocs.length;
    
    clusterDocs.forEach(doc => {
      Object.entries(doc.terms).forEach(([term, freq]) => {
        clusterTermFreq[term] = (clusterTermFreq[term] || 0) + freq;
      });
    });

    const scores = Object.entries(clusterTermFreq).map(([term, clusterFreq]) => {
      const tf = clusterFreq / clusterSize;
      const docsWithTerm = allDocs.filter(doc => doc.terms[term] > 0).length;
      const idf = Math.log(allDocs.length / (docsWithTerm || 1));
      
      let boost = 1;
      if (term.includes(' ')) boost *= 2.0;
      if (term.length > 7 && !term.includes(' ')) boost *= 1.3;
      if (/^(research|analysis|study|report|management|development|strategy|implementation|assessment|evaluation|methodology|framework|approach|technology|innovation)/.test(term)) {
        boost *= 1.8;
      }
      
      return { 
        term,
        score: tf * idf * boost
      };
    });

    const topTerms = scores
      .filter(item => item.term.length >= 4 && item.score > 0.01)
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);

    if (topTerms.length === 0) return "Miscellaneous Documents";

    const formatTerm = (term) => {
      return term.split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
    };

    const formattedTerms = topTerms.map(item => formatTerm(item.term));

    if (formattedTerms.length === 1) {
      return clusterSize >= 5 ? `${formattedTerms[0]} Collection` : formattedTerms[0];
    } else if (formattedTerms.length === 2) {
      return `${formattedTerms[0]} & ${formattedTerms[1]}`;
    } else {
      return `${formattedTerms[0]}, ${formattedTerms[1]} & ${formattedTerms[2]}`;
    }
  };

  // Enhanced sunburst visualization with sophisticated styling
  const createSunburst = useCallback((data) => {
    if (!data || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 900;
    const height = 900;
    const radius = Math.min(width, height) / 2 - 80;

    // Sophisticated color scales for different levels
    const primaryColors = ['#2563eb', '#059669', '#dc2626', '#7c3aed', '#ea580c', '#0891b2', '#be185d', '#4338ca'];
    const getColor = (d) => {
      if (d.depth === 0) return '#f8f9fa';
      
      const rootParent = d.ancestors().find(ancestor => ancestor.depth === 1);
      const baseColorIndex = rootParent ? rootParent.data.id.charCodeAt(rootParent.data.id.length - 1) % primaryColors.length : 0;
      const baseColor = d3.color(primaryColors[baseColorIndex]);
      
      if (d.depth === 1) return baseColor.toString();
      if (d.depth === 2) return baseColor.brighter(0.4).toString();
      return baseColor.brighter(0.7).toString();
    };

    const partition = d3.partition()
      .size([2 * Math.PI, radius])
      .padding(0.002);

    const arc = d3.arc()
      .startAngle(d => d.x0)
      .endAngle(d => d.x1)
      .innerRadius(d => Math.max(0, d.y0))
      .outerRadius(d => Math.max(d.y0, d.y1));

    svg
      .attr("width", width)
      .attr("height", height);

    const g = svg.append("g")
      .attr("transform", `translate(${width / 2},${height / 2})`);

    const root = d3.hierarchy(data)
      .sum(d => d.children ? 0 : d.size)
      .sort((a, b) => b.value - a.value);

    partition(root);

    // Add circular grid lines
    const gridRadii = [radius * 0.3, radius * 0.6, radius * 0.9];
    g.selectAll(".grid-circle")
      .data(gridRadii)
      .enter().append("circle")
      .attr("class", "grid-circle")
      .attr("r", d => d)
      .style("fill", "none")
      .style("stroke", "#e5e7eb")
      .style("stroke-width", 1)
      .style("opacity", 0.3);

    // Add radial grid lines
    const angleStep = Math.PI / 6; // 30 degrees
    for (let i = 0; i < 12; i++) {
      const angle = i * angleStep;
      g.append("line")
        .attr("x1", 0)
        .attr("y1", 0)
        .attr("x2", Math.cos(angle - Math.PI/2) * radius)
        .attr("y2", Math.sin(angle - Math.PI/2) * radius)
        .style("stroke", "#e5e7eb")
        .style("stroke-width", 1)
        .style("opacity", 0.2);
    }

    // Add numerical scale around the perimeter
    const maxValue = d3.max(root.descendants(), d => d.value);
    const scaleSteps = 8;
    for (let i = 0; i <= scaleSteps; i++) {
      const angle = (i / scaleSteps) * 2 * Math.PI - Math.PI/2;
      const value = Math.round((i / scaleSteps) * maxValue);
      const textRadius = radius + 25;
      
      g.append("text")
        .attr("x", Math.cos(angle) * textRadius)
        .attr("y", Math.sin(angle) * textRadius)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .style("font-size", "11px")
        .style("font-weight", "500")
        .style("fill", "#6b7280")
        .text(value.toLocaleString());
      
      // Add tick marks
      g.append("line")
        .attr("x1", Math.cos(angle) * (radius - 5))
        .attr("y1", Math.sin(angle) * (radius - 5))
        .attr("x2", Math.cos(angle) * (radius + 10))
        .attr("y2", Math.sin(angle) * (radius + 10))
        .style("stroke", "#9ca3af")
        .style("stroke-width", 2);
    }

    // Create the sunburst segments
    const paths = g.selectAll("path")
      .data(root.descendants())
      .enter().append("path")
      .attr("d", arc)
      .style("fill", getColor)
      .style("stroke", "#ffffff")
      .style("stroke-width", d => d.depth === 0 ? 0 : 1.5)
      .style("opacity", d => d.depth === 0 ? 0 : 0.9)
      .style("cursor", d => d.depth > 0 ? "pointer" : "default");

    // Add sophisticated hover and click interactions
    paths
      .on("mouseover", function(event, d) {
        if (d.depth === 0) return;
        
        // Highlight the hovered segment and its ancestors
        paths.style("opacity", 0.3);
        d.ancestors().forEach(ancestor => {
          paths.filter(p => p === ancestor).style("opacity", 1);
        });
        
        // Enhanced tooltip
        const tooltip = g.append("g").attr("class", "tooltip");
        
        tooltip.append("rect")
          .attr("x", -180)
          .attr("y", -80)
          .attr("width", 360)
          .attr("height", 160)
          .attr("rx", 12)
          .style("fill", "rgba(15, 23, 42, 0.95)")
          .style("stroke", "#334155")
          .style("stroke-width", 1)
          .style("filter", "drop-shadow(0 10px 25px rgba(0,0,0,0.5))");
        
        const text = tooltip.append("text")
          .attr("text-anchor", "middle")
          .style("fill", "white");
        
        // Title
        text.append("tspan")
          .attr("x", 0)
          .attr("dy", -40)
          .style("font-size", "16px")
          .style("font-weight", "700")
          .style("fill", "#ffffff")
          .text(d.data.name.length > 35 ? d.data.name.substring(0, 32) + "..." : d.data.name);
        
        // Statistics
        if (d.data.documents) {
          text.append("tspan")
            .attr("x", 0)
            .attr("dy", 25)
            .style("font-size", "14px")
            .style("font-weight", "500")
            .style("fill", "#94a3b8")
            .text(`${d.data.documents.length} document${d.data.documents.length > 1 ? 's' : ''} • Level ${d.data.level || d.depth}`);
          
          text.append("tspan")
            .attr("x", 0)
            .attr("dy", 20)
            .style("font-size", "12px")
            .style("fill", "#64748b")
            .text(`Total terms: ${d.value.toLocaleString()}`);
          
          // Show percentage of total
          const percentage = ((d.value / root.value) * 100).toFixed(1);
          text.append("tspan")
            .attr("x", 0)
            .attr("dy", 18)
            .style("font-size", "12px")
            .style("fill", "#64748b")
            .text(`${percentage}% of collection`);
          
          // Show sample terms for clusters
          if (d.data.documents.length > 1) {
            const sampleTerms = d.data.documents
              .flatMap(doc => Object.keys(doc.terms))
              .slice(0, 4)
              .join(', ');
            
            text.append("tspan")
              .attr("x", 0)
              .attr("dy", 20)
              .style("font-size", "11px")
              .style("fill", "#475569")
              .style("font-style", "italic")
              .text(`Keywords: ${sampleTerms}`);
          }
        }
      })
      .on("mouseout", function(event, d) {
        if (d.depth === 0) return;
        paths.style("opacity", p => p.depth === 0 ? 0 : 0.9);
        g.selectAll(".tooltip").remove();
      })
      .on("click", function(event, d) {
        if (d.depth === 0) return;
        event.stopPropagation();
        
        if (d.data.documents) {
          setSelectedCluster({
            name: d.data.name,
            documents: d.data.documents,
            level: d.data.level || d.depth
          });
        }
        
        // Visual selection feedback
        paths.style("stroke-width", p => p.depth === 0 ? 0 : 1.5);
        d3.select(this).style("stroke-width", 4).style("stroke", "#1f2937");
      });

    // Add sophisticated center labels
    const centerGroup = g.append("g").attr("class", "center-labels");
    
    // Main title
    centerGroup.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "-1em")
      .style("font-size", "24px")
      .style("font-weight", "800")
      .style("fill", "#1e293b")
      .text("Topics");
    
    // Document count
    centerGroup.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "0.3em")
      .style("font-size", "16px")
      .style("font-weight", "600")
      .style("fill", "#475569")
      .text(`${documents.length} Documents`);
    
    // Cluster count
    if (root.children && root.children.length > 0) {
      centerGroup.append("text")
        .attr("text-anchor", "middle")
        .attr("dy", "1.8em")
        .style("font-size", "14px")
        .style("font-weight", "500")
        .style("fill", "#64748b")
        .text(`${root.children.length} Main ${root.children.length === 1 ? 'Cluster' : 'Clusters'}`);
    }

    // Add sophisticated labels for main segments
    root.children?.forEach(d => {
      const angle = (d.x0 + d.x1) / 2;
      const labelRadius = (d.y0 + d.y1) / 2 + 15;
      const rotation = angle * 180 / Math.PI - 90;
      const isRightSide = angle < Math.PI;
      
      if (d.x1 - d.x0 > 0.1) { // Only label large enough segments
        const labelGroup = g.append("g")
          .attr("transform", `rotate(${rotation}) translate(${labelRadius},0) rotate(${isRightSide ? 0 : 180})`);
        
        // Main cluster name
        labelGroup.append("text")
          .attr("dy", "0.35em")
          .attr("text-anchor", isRightSide ? "start" : "end")
          .style("font-size", "13px")
          .style("font-weight", "700")
          .style("fill", "#1e293b")
          .style("text-shadow", "1px 1px 2px rgba(255,255,255,0.8)")
          .text(d.data.name.length > 20 ? d.data.name.substring(0, 17) + "..." : d.data.name);
        
        // Document count
        if (d.data.documents) {
          labelGroup.append("text")
            .attr("dy", "2em")
            .attr("text-anchor", isRightSide ? "start" : "end")
            .style("font-size", "11px")
            .style("font-weight", "500")
            .style("fill", "#475569")
            .style("text-shadow", "1px 1px 1px rgba(255,255,255,0.8)")
            .text(`${d.data.documents.length} docs`);
        }
      }
    });

    // Add a subtle border around the entire chart
    g.append("circle")
      .attr("r", radius + 1)
      .style("fill", "none")
      .style("stroke", "#d1d5db")
      .style("stroke-width", 2);

  }, [documents]);

  // File processing with hierarchical clustering
  const processFiles = async () => {
    if (selectedFiles.length === 0) return;

    setIsProcessing(true);
    const processedDocs = [];

    for (const file of selectedFiles) {
      try {
        let sections;
        const extension = file.name.split('.').pop().toLowerCase();

        if (extension === 'txt') {
          sections = await parseTextFile(file);
        } else if (extension === 'docx' || extension === 'doc') {
          sections = await parseWordDoc(file);
        } else if (extension === 'pdf') {
          sections = await parsePDF(file);
        } else {
          sections = await parseTextFile(file);
        }

        const combinedText = [sections.title, sections.content].filter(Boolean).join(' ');

        if (combinedText.length < 50) {
          console.warn(`Skipping ${file.name}: insufficient content`);
          continue;
        }

        const terms = extractTerms(combinedText);
        if (Object.keys(terms).length === 0) {
          console.warn(`Skipping ${file.name}: no meaningful terms extracted`);
          continue;
        }

        processedDocs.push({
          name: file.name,
          title: sections.title,
          content: combinedText,
          terms,
          size: file.size
        });
      } catch (error) {
        console.error(`Error processing ${file.name}:`, error);
      }
    }

    setDocuments(processedDocs);
    
    if (processedDocs.length > 0) {
      const embeddings = createTFIDFEmbeddings(processedDocs);
      const hierarchyResult = hierarchicalClustering(embeddings, processedDocs);
      setClustering(hierarchyResult);
    } else {
      setClustering(null);
    }
    
    setIsProcessing(false);
  };

  useEffect(() => {
    if (clustering) {
      createSunburst(clustering);
    }
  }, [clustering, createSunburst]);

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
  };

  const removeFile = (index) => {
    const newFiles = selectedFiles.filter((_, i) => i !== index);
    setSelectedFiles(newFiles);
  };

  return (
    <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      <div style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1 style={{ fontSize: '2.5rem', fontWeight: '700', color: '#1f2937', marginBottom: '10px' }}>
          Enhanced Hierarchical Document Clustering
        </h1>
        <p style={{ fontSize: '1.1rem', color: '#6b7280' }}>
          Advanced document clustering with automatic hierarchical topic subdivision
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', marginBottom: '30px' }}>
        {/* File Upload Section */}
        <div style={{ background: 'white', padding: '25px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#1f2937', marginBottom: '20px' }}>
            Upload Documents
          </h2>
          
          <div style={{ marginBottom: '20px' }}>
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
                Select Files
              </button>
              <button
                onClick={processFiles}
                disabled={selectedFiles.length === 0 || isProcessing}
                style={{
                  padding: '12px 20px',
                  backgroundColor: selectedFiles.length === 0 || isProcessing ? '#9ca3af' : '#10b981',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: selectedFiles.length === 0 || isProcessing ? 'not-allowed' : 'pointer',
                  fontWeight: '500'
                }}
              >
                {isProcessing ? 'Processing...' : 'Analyze Documents'}
              </button>
            </div>
          </div>

          {selectedFiles.length > 0 && (
            <div style={{ marginBottom: '20px' }}>
              <h3 style={{ fontSize: '1.2rem', fontWeight: '500', marginBottom: '10px' }}>
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
                      <div style={{ fontSize: '0.875rem', color: '#6b7280' }}>
                        {(file.size / 1024).toFixed(1)} KB
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
                        cursor: 'pointer',
                        fontSize: '14px'
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
              <h3 style={{ fontSize: '1.2rem', fontWeight: '500', marginBottom: '10px' }}>
                Processed Documents ({documents.length}):
              </h3>
              <div style={{ maxHeight: '300px', overflowY: 'auto', border: '1px solid #e5e7eb', borderRadius: '6px' }}>
                {documents.map((doc, index) => (
                  <div key={index} style={{ 
                    padding: '12px',
                    borderBottom: index < documents.length - 1 ? '1px solid #f3f4f6' : 'none'
                  }}>
                    <div style={{ fontWeight: '500', marginBottom: '4px' }}>{doc.name}</div>
                    <div style={{ fontSize: '0.875rem', color: '#6b7280', marginBottom: '4px' }}>
                      {Object.keys(doc.terms).length} unique terms
                    </div>
                    <div style={{ fontSize: '0.8rem', color: '#9ca3af' }}>
                      Key terms: {Object.entries(doc.terms)
                        .sort(([,a], [,b]) => b - a)
                        .slice(0, 4)
                        .map(([term]) => term)
                        .join(', ')}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Visualization Section */}
        <div style={{ background: 'white', padding: '25px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#1f2937' }}>
              Topic Hierarchy
            </h2>
          </div>
          
          {clustering ? (
            <div style={{ textAlign: 'center' }}>
              <svg ref={svgRef}></svg>
            </div>
          ) : (
            <div style={{ 
              textAlign: 'center', 
              padding: '60px 20px',
              border: '2px dashed #d1d5db',
              borderRadius: '8px',
              color: '#6b7280'
            }}>
              <div style={{ fontSize: '3rem', marginBottom: '16px' }}>🎯</div>
              <p style={{ fontSize: '1.1rem', fontWeight: '500', marginBottom: '8px' }}>
                Upload documents to see hierarchical topic clusters
              </p>
              <p style={{ fontSize: '0.9rem' }}>
                Large topics will be automatically subdivided for better organization
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
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h3 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#1f2937' }}>
              📁 {selectedCluster.name}
            </h3>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <span style={{
                padding: '4px 12px',
                backgroundColor: '#dbeafe',
                color: '#1e40af',
                borderRadius: '16px',
                fontSize: '0.875rem',
                fontWeight: '500'
              }}>
                Level {selectedCluster.level} • {selectedCluster.documents.length} document{selectedCluster.documents.length > 1 ? 's' : ''}
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
                  fontSize: '18px'
                }}
              >
                ✕
              </button>
            </div>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '20px' }}>
            {selectedCluster.documents.map((doc, index) => (
              <div key={index} style={{ 
                border: '1px solid #e5e7eb', 
                borderRadius: '8px', 
                padding: '16px',
                backgroundColor: '#f9fafb'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '12px' }}>
                  <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#1f2937', margin: 0 }}>
                    📄 {doc.name}
                  </h4>
                  <div style={{ display: 'flex', gap: '8px', fontSize: '0.8rem', color: '#6b7280' }}>
                    <span>{(doc.size / 1024).toFixed(1)} KB</span>
                    <span>{Object.keys(doc.terms).length} terms</span>
                  </div>
                </div>
                
                {doc.content && (
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ fontSize: '0.9rem', color: '#4b5563', lineHeight: '1.5' }}>
                      {doc.content.length > 200 ? doc.content.substring(0, 200) + "..." : doc.content}
                    </div>
                  </div>
                )}
                
                <div>
                  <h5 style={{ fontSize: '0.875rem', fontWeight: '600', color: '#374151', marginBottom: '8px' }}>
                    Key Terms:
                  </h5>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                    {Object.entries(doc.terms)
                      .sort(([,a], [,b]) => b - a)
                      .slice(0, 8)
                      .map(([term, freq]) => (
                        <span key={term} style={{
                          padding: '4px 8px',
                          backgroundColor: '#e0e7ff',
                          color: '#3730a3',
                          borderRadius: '12px',
                          fontSize: '0.75rem',
                          fontWeight: '500'
                        }}>
                          {term} ({freq})
                        </span>
                      ))}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {selectedCluster.documents.length > 1 && (
            <div style={{ 
              marginTop: '20px', 
              padding: '16px', 
              backgroundColor: '#f0f9ff', 
              borderRadius: '8px',
              border: '1px solid #bae6fd'
            }}>
              <h4 style={{ fontSize: '1rem', fontWeight: '600', color: '#0c4a6e', marginBottom: '8px' }}>
                📊 Cluster Summary
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
                <div style={{ fontSize: '0.875rem' }}>
                  <span style={{ fontWeight: '500', color: '#374151' }}>Total Documents:</span>{' '}
                  <span style={{ color: '#0c4a6e' }}>{selectedCluster.documents.length}</span>
                </div>
                <div style={{ fontSize: '0.875rem' }}>
                  <span style={{ fontWeight: '500', color: '#374151' }}>Total Size:</span>{' '}
                  <span style={{ color: '#0c4a6e' }}>
                    {(selectedCluster.documents.reduce((sum, doc) => sum + doc.size, 0) / 1024).toFixed(1)} KB
                  </span>
                </div>
                <div style={{ fontSize: '0.875rem' }}>
                  <span style={{ fontWeight: '500', color: '#374151' }}>Unique Terms:</span>{' '}
                  <span style={{ color: '#0c4a6e' }}>
                    {new Set(selectedCluster.documents.flatMap(doc => Object.keys(doc.terms))).size}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Instructions */}
      <div style={{ 
        background: 'white', 
        padding: '25px', 
        borderRadius: '12px', 
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
      }}>
        <h3 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#1f2937', marginBottom: '20px' }}>
          How Enhanced Hierarchical Clustering Works:
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
          <div>
            <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#1f2937', marginBottom: '8px' }}>
              📄 Advanced Text Processing:
            </h4>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: '#4b5563' }}>
              <li style={{ marginBottom: '4px' }}>• Aggressive metadata and noise removal</li>
              <li style={{ marginBottom: '4px' }}>• POS-tagging inspired term extraction</li>
              <li style={{ marginBottom: '4px' }}>• TF-IDF embeddings with n-grams</li>
            </ul>
          </div>
          <div>
            <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#1f2937', marginBottom: '8px' }}>
              🎯 Hierarchical Clustering:
            </h4>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: '#4b5563' }}>
              <li style={{ marginBottom: '4px' }}>• HDBSCAN-inspired density clustering</li>
              <li style={{ marginBottom: '4px' }}>• Automatic subdivision of large clusters</li>
              <li style={{ marginBottom: '4px' }}>• Multi-level topic organization</li>
            </ul>
          </div>
          <div>
            <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#1f2937', marginBottom: '8px' }}>
              🏷️ Smart Topic Naming:
            </h4>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0, color: '#4b5563' }}>
              <li style={{ marginBottom: '4px' }}>• c-TF-IDF based term scoring</li>
              <li style={{ marginBottom: '4px' }}>• Context-aware name generation</li>
              <li style={{ marginBottom: '4px' }}>• Hierarchical topic relationships</li>
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
            💡 <strong>Tip:</strong> This system automatically organizes large topic clusters into sub-topics, 
            making it ideal for analyzing large document collections. Upload 10+ documents covering related 
            themes for best results.
          </p>
        </div>
      </div>
    </div>
  );
};

export default EnhancedDocumentClusteringApp;