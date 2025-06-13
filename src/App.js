import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as d3 from 'd3';
import * as mammoth from 'mammoth';
import _ from 'lodash';
import './App.css';

const DocumentClusteringApp = () => {
  const [documents, setDocuments] = useState([]);
  const [clustering, setClustering] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const svgRef = useRef(null);
  const fileInputRef = useRef(null);

  // Enhanced text processing utilities
  const preprocessText = (text) => {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  };

  const extractTerms = (text) => {
    const stopWords = new Set([
      'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
      'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
      'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
      'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
      'his', 'its', 'our', 'their', 'from', 'up', 'out', 'down', 'off', 'over', 'under', 'again',
      'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
      'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
      'own', 'same', 'so', 'than', 'too', 'very', 'can', 'just', 'should', 'now', 'also', 'one',
      'two', 'first', 'second', 'new', 'old', 'good', 'great', 'small', 'large', 'big', 'little',
      'get', 'make', 'go', 'know', 'take', 'see', 'come', 'think', 'look', 'want', 'give', 'use',
      'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'keep', 'let', 'begin'
    ]);

    // Extract n-grams (1-grams and 2-grams)
    const words = preprocessText(text).split(/\s+/).filter(word => word.length > 2);
    const terms = {};

    // Add unigrams
    words.forEach(word => {
      if (!stopWords.has(word)) {
        terms[word] = (terms[word] || 0) + 1;
      }
    });

    // Add bigrams
    for (let i = 0; i < words.length - 1; i++) {
      const bigram = `${words[i]} ${words[i + 1]}`;
      if (!stopWords.has(words[i]) && !stopWords.has(words[i + 1])) {
        terms[bigram] = (terms[bigram] || 0) + 1;
      }
    }

    return terms;
  };

  // Document parsers
  const parseTextFile = (file) => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.readAsText(file);
    });
  };

  const parseWordDoc = async (file) => {
    try {
      const arrayBuffer = await file.arrayBuffer();
      const result = await mammoth.extractRawText({ arrayBuffer });
      return result.value;
    } catch (error) {
      console.error('Error parsing Word document:', error);
      return '';
    }
  };

  const parsePDF = async (file) => {
    try {
      const text = await file.text();
      return text.replace(/[^\x20-\x7E]/g, ' ').replace(/\s+/g, ' ');
    } catch (error) {
      console.error('Error parsing PDF:', error);
      return '';
    }
  };

  // TF-IDF based document embeddings
  const createTFIDFEmbeddings = (documents) => {
    // Build vocabulary
    const termCounts = {};
    const docCount = documents.length;

    documents.forEach(doc => {
      const uniqueTerms = new Set(Object.keys(doc.terms));
      uniqueTerms.forEach(term => {
        termCounts[term] = (termCounts[term] || 0) + 1;
      });
    });

    // Filter vocabulary (remove very rare and very common terms)
    const vocabulary = Object.entries(termCounts)
      .filter(([term, count]) => count >= Math.max(1, Math.floor(docCount * 0.05)) && count <= Math.floor(docCount * 0.8))
      .sort(([,a], [,b]) => b - a)
      .slice(0, 500) // Limit to top 500 terms
      .map(([term]) => term);

    // Create TF-IDF vectors
    return documents.map(doc => {
      const vector = new Array(vocabulary.length).fill(0);
      const totalTerms = Object.values(doc.terms).reduce((sum, count) => sum + count, 0);

      vocabulary.forEach((term, index) => {
        const tf = (doc.terms[term] || 0) / totalTerms;
        const idf = Math.log(docCount / (termCounts[term] || 1));
        vector[index] = tf * idf;
      });

      // Normalize vector
      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      return magnitude > 0 ? vector.map(val => val / magnitude) : vector;
    });
  };

  // DBSCAN clustering implementation
  const dbscanClustering = (embeddings, eps = 0.4, minPts = 2) => {
    const n = embeddings.length;
    const visited = new Array(n).fill(false);
    const clusters = new Array(n).fill(-1); // -1 means noise
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
      
      if (neighbors.length < minPts - 1) { // -1 because we don't count the point itself
        continue;
      }
      
      // Start new cluster
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

  // Generate cluster names using c-TF-IDF approach
  const generateClusterName = (clusterDocs, allDocs) => {
    if (clusterDocs.length === 0) return "Empty Cluster";
    if (clusterDocs.length === 1) {
      const name = clusterDocs[0].name.replace(/\.[^/.]+$/, "");
      return name.length > 20 ? name.substring(0, 17) + "..." : name;
    }

    // Calculate c-TF-IDF scores
    const clusterTermFreq = {};
    const clusterSize = clusterDocs.length;
    
    // Aggregate terms from cluster documents
    clusterDocs.forEach(doc => {
      Object.entries(doc.terms).forEach(([term, freq]) => {
        clusterTermFreq[term] = (clusterTermFreq[term] || 0) + freq;
      });
    });

    // Calculate c-TF-IDF for each term
    const scores = Object.entries(clusterTermFreq).map(([term, clusterFreq]) => {
      // Term frequency in cluster
      const tf = clusterFreq / clusterSize;
      
      // How many documents across ALL clusters contain this term
      const docsWithTerm = allDocs.filter(doc => doc.terms[term] > 0).length;
      const idf = Math.log(allDocs.length / (docsWithTerm || 1));
      
      return { term, score: tf * idf };
    });

    // Get top terms, prioritizing meaningful phrases
    const topTerms = scores
      .filter(item => item.term.length > 2) // Filter out very short terms
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);

    // Prefer bigrams for more descriptive names
    const bigrams = topTerms.filter(item => item.term.includes(' '));
    const unigrams = topTerms.filter(item => !item.term.includes(' '));

    let selectedTerms = [];
    if (bigrams.length > 0) {
      selectedTerms.push(bigrams[0].term);
      if (bigrams.length > 1) {
        selectedTerms.push(bigrams[1].term);
      } else if (unigrams.length > 0) {
        selectedTerms.push(unigrams[0].term);
      }
    } else if (unigrams.length >= 2) {
      selectedTerms = unigrams.slice(0, 2).map(item => item.term);
    } else if (unigrams.length === 1) {
      selectedTerms = [unigrams[0].term];
    }

    // Generate descriptive name
    if (selectedTerms.length >= 2) {
      return selectedTerms.map(term => 
        term.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')
      ).join(' & ');
    } else if (selectedTerms.length === 1) {
      const term = selectedTerms[0];
      const formatted = term.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
      return `${formatted} Topic`;
    } else {
      return 'Mixed Documents';
    }
  };

  // BERTopic-inspired clustering pipeline
  const bertopicClustering = async (documents) => {
    if (documents.length === 0) return null;
    if (documents.length === 1) {
      return {
        id: 'single_doc',
        name: documents[0].name.replace(/\.[^/.]+$/, ""),
        size: Object.values(documents[0].terms).reduce((a, b) => a + b, 0),
        children: null,
        documents: documents
      };
    }

    // Step 1: Create TF-IDF embeddings
    const embeddings = createTFIDFEmbeddings(documents);

    // Step 2: DBSCAN clustering with adaptive parameters
    let eps = 0.3;
    let minPts = Math.max(2, Math.floor(documents.length * 0.1));
    let clusters = dbscanClustering(embeddings, eps, minPts);

    // Adjust parameters if too many noise points
    const noiseCount = clusters.filter(label => label === -1).length;
    if (noiseCount > documents.length * 0.5) {
      eps = 0.2;
      minPts = Math.max(2, Math.floor(documents.length * 0.05));
      clusters = dbscanClustering(embeddings, eps, minPts);
    }

    // Step 3: Build hierarchical structure with named clusters
    return buildNamedClusters(documents, clusters);
  };

  // Build named hierarchical clusters
  const buildNamedClusters = (documents, clusterLabels) => {
    const clusterMap = new Map();
    const noiseDocs = [];

    // Group documents by cluster
    documents.forEach((doc, index) => {
      const clusterId = clusterLabels[index];
      if (clusterId === -1) {
        noiseDocs.push(doc);
      } else {
        if (!clusterMap.has(clusterId)) {
          clusterMap.set(clusterId, []);
        }
        clusterMap.get(clusterId).push(doc);
      }
    });

    // Create cluster nodes with names
    const clusters = Array.from(clusterMap.entries()).map(([clusterId, clusterDocs]) => ({
      id: `cluster_${clusterId}`,
      name: generateClusterName(clusterDocs, documents),
      size: clusterDocs.reduce((sum, doc) => sum + Object.values(doc.terms).reduce((a, b) => a + b, 0), 0),
      children: clusterDocs.map(doc => ({
        id: `doc_${doc.name}`,
        name: doc.name.replace(/\.[^/.]+$/, ""),
        size: Object.values(doc.terms).reduce((a, b) => a + b, 0),
        children: null,
        documents: [doc]
      })),
      documents: clusterDocs
    }));

    // Add noise documents as individual clusters
    noiseDocs.forEach((doc, index) => {
      clusters.push({
        id: `outlier_${index}`,
        name: doc.name.replace(/\.[^/.]+$/, ""),
        size: Object.values(doc.terms).reduce((a, b) => a + b, 0),
        children: null,
        documents: [doc]
      });
    });

    // Create root node
    if (clusters.length === 0) return null;
    if (clusters.length === 1) return clusters[0];

    return {
      id: 'root',
      name: 'Document Collection',
      size: clusters.reduce((sum, cluster) => sum + cluster.size, 0),
      children: clusters,
      documents: documents
    };
  };

  // Enhanced sunburst visualization
  const createSunburst = useCallback((data) => {
    if (!data || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 700;
    const height = 700;
    const radius = Math.min(width, height) / 2 - 10;

    // Color scheme
    const colorScale = d3.scaleOrdinal()
      .range(['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD', '#FF7675', '#00B894']);

    const partition = d3.partition()
      .size([2 * Math.PI, radius])
      .padding(0.003);

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

    // Create paths with enhanced styling
    g.selectAll("path")
      .data(root.descendants())
      .enter().append("path")
      .attr("d", arc)
      .style("fill", (d, i) => {
        if (d.depth === 0) return '#f8f9fa';
        if (d.depth === 1) return colorScale(d.data.id);
        return d3.color(colorScale(d.parent.data.id)).brighter(0.3);
      })
      .style("stroke", "#fff")
      .style("stroke-width", d => d.depth === 0 ? 0 : 2)
      .style("opacity", d => d.depth === 0 ? 0.1 : 0.85)
      .style("cursor", d => d.depth > 0 ? "pointer" : "default")
      .on("mouseover", function(event, d) {
        if (d.depth === 0) return;
        
        d3.select(this)
          .style("opacity", 1)
          .style("stroke-width", 3);
        
        // Enhanced tooltip with document previews
        const tooltip = g.append("g")
          .attr("class", "tooltip");
        
        // Determine tooltip content and size
        let tooltipHeight = 70;
        let tooltipWidth = 220;
        
        // For individual documents, show preview
        if (d.depth === 2 || (d.depth === 1 && d.data.documents && d.data.documents.length === 1)) {
          tooltipHeight = 140;
          tooltipWidth = 280;
        } else if (d.depth === 1 && d.data.documents && d.data.documents.length > 1) {
          tooltipHeight = 120;
          tooltipWidth = 260;
        }
        
        tooltip.append("rect")
          .attr("x", -tooltipWidth/2)
          .attr("y", -tooltipHeight/2)
          .attr("width", tooltipWidth)
          .attr("height", tooltipHeight)
          .attr("rx", 8)
          .style("fill", "rgba(0,0,0,0.9)")
          .style("stroke", "#fff")
          .style("stroke-width", 1)
          .style("filter", "drop-shadow(0 4px 12px rgba(0,0,0,0.3))");
        
        const text = tooltip.append("text")
          .attr("text-anchor", "middle")
          .style("fill", "white")
          .style("font-size", "12px");
        
        // Title
        text.append("tspan")
          .attr("x", 0)
          .attr("dy", -tooltipHeight/2 + 20)
          .style("font-weight", "bold")
          .style("font-size", "13px")
          .text(d.data.name.length > 30 ? d.data.name.substring(0, 27) + "..." : d.data.name);
        
        if (d.data.documents) {
          // Document count
          text.append("tspan")
            .attr("x", 0)
            .attr("dy", "18")
            .style("font-size", "11px")
            .style("fill", "#bbb")
            .text(`${d.data.documents.length} document${d.data.documents.length > 1 ? 's' : ''}`);
          
          // For single document (individual arc or single-doc cluster)
          if (d.data.documents.length === 1) {
            const doc = d.data.documents[0];
            
            // Show document preview
            const preview = doc.text.length > 180 ? 
              doc.text.substring(0, 180) + "..." : 
              doc.text;
            
            // Split preview into lines
            const words = preview.split(' ');
            let currentLine = '';
            let lineY = 20;
            
            for (let i = 0; i < words.length; i++) {
              const testLine = currentLine + words[i] + ' ';
              if (testLine.length > 35 && currentLine !== '') {
                text.append("tspan")
                  .attr("x", 0)
                  .attr("dy", lineY)
                  .style("font-size", "10px")
                  .style("fill", "#ddd")
                  .text(currentLine.trim());
                currentLine = words[i] + ' ';
                lineY = 14;
              } else {
                currentLine = testLine;
              }
              
              // Limit to 5 lines
              if (lineY > 70) break;
            }
            
            // Add remaining text
            if (currentLine.trim() && lineY <= 70) {
              text.append("tspan")
                .attr("x", 0)
                .attr("dy", lineY)
                .style("font-size", "10px")
                .style("fill", "#ddd")
                .text(currentLine.trim());
            }
            
            // Show top terms
            const topTerms = Object.entries(doc.terms)
              .sort(([,a], [,b]) => b - a)
              .slice(0, 4)
              .map(([term]) => term);
            
            text.append("tspan")
              .attr("x", 0)
              .attr("dy", "20")
              .style("font-size", "9px")
              .style("fill", "#999")
              .text(`Terms: ${topTerms.join(', ')}`);
            
          } else if (d.depth === 1) {
            // For clusters with multiple documents
            const topTerms = _.chain(d.data.documents)
              .flatMap(doc => Object.entries(doc.terms))
              .groupBy(([term]) => term)
              .mapValues(group => _.sumBy(group, ([, freq]) => freq))
              .toPairs()
              .orderBy(([, freq]) => freq, 'desc')
              .take(5)
              .map(([term]) => term)
              .value();
            
            text.append("tspan")
              .attr("x", 0)
              .attr("dy", "20")
              .style("font-size", "10px")
              .style("fill", "#ccc")
              .text(`Key terms: ${topTerms.join(', ')}`);
            
            // Show sample document names
            const sampleDocs = d.data.documents.slice(0, 3).map(doc => 
              doc.name.replace(/\.[^/.]+$/, "").substring(0, 25)
            );
            
            text.append("tspan")
              .attr("x", 0)
              .attr("dy", "16")
              .style("font-size", "9px")
              .style("fill", "#aaa")
              .text(`Files: ${sampleDocs.join(', ')}${d.data.documents.length > 3 ? '...' : ''}`);
            
            // Show a preview from the first document
            if (d.data.documents[0] && d.data.documents[0].text) {
              const preview = d.data.documents[0].text.length > 120 ? 
                d.data.documents[0].text.substring(0, 120) + "..." : 
                d.data.documents[0].text;
              
              text.append("tspan")
                .attr("x", 0)
                .attr("dy", "16")
                .style("font-size", "9px")
                .style("fill", "#888")
                .style("font-style", "italic")
                .text(`"${preview}"`);
            }
          }
        }
      })
      .on("mouseout", function(event, d) {
        if (d.depth === 0) return;
        
        d3.select(this)
          .style("opacity", 0.85)
          .style("stroke-width", 2);
        
        g.selectAll(".tooltip").remove();
      });

    // Add enhanced center label with clustering info
    const centerGroup = g.append("g").attr("class", "center-labels");
    
    centerGroup.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "-0.3em")
      .style("font-size", "18px")
      .style("font-weight", "bold")
      .style("fill", "#1f2937")
      .text(`${documents.length} Documents`);
    
    if (root.children && root.children.length > 1) {
      centerGroup.append("text")
        .attr("text-anchor", "middle")
        .attr("dy", "1.2em")
        .style("font-size", "12px")
        .style("font-weight", "500")
        .style("fill", "#6b7280")
        .text(`${root.children.length} Topics`);
    } else if (root.children && root.children.length === 1) {
      centerGroup.append("text")
        .attr("text-anchor", "middle")
        .attr("dy", "1.2em")
        .style("font-size", "12px")
        .style("font-weight", "500")
        .style("fill", "#6b7280")
        .text("Single Topic");
    }

    // Add enhanced cluster labels with better positioning and content
    const clusterLabels = g.selectAll("g.cluster-label-group")
      .data(root.children || [])
      .enter().append("g")
      .attr("class", "cluster-label-group");

    clusterLabels.each(function(d) {
      const labelGroup = d3.select(this);
      const angle = (d.x0 + d.x1) / 2;
      const radius = (d.y0 + d.y1) / 2;
      const rotation = angle * 180 / Math.PI - 90;
      const isRightSide = angle < Math.PI;
      
      // Calculate optimal label position
      const labelRadius = Math.max(radius + 20, d.y1 + 15);
      
      labelGroup.attr("transform", `rotate(${rotation}) translate(${labelRadius},0) rotate(${isRightSide ? 0 : 180})`);
      
      // Main cluster name
      const clusterName = d.data.name;
      let displayName = clusterName;
      
      // Smart truncation - try to keep meaningful words
      if (clusterName.length > 25) {
        const words = clusterName.split(' ');
        if (words.length > 1) {
          // Try to keep first meaningful words
          let truncated = words[0];
          for (let i = 1; i < words.length; i++) {
            if (truncated.length + words[i].length + 1 <= 22) {
              truncated += ' ' + words[i];
            } else {
              truncated += '...';
              break;
            }
          }
          displayName = truncated;
        } else {
          displayName = clusterName.substring(0, 22) + '...';
        }
      }
      
      labelGroup.append("text")
        .attr("dy", "0.35em")
        .attr("text-anchor", isRightSide ? "start" : "end")
        .style("font-size", "11px")
        .style("font-weight", "700")
        .style("fill", "#1f2937")
        .style("text-shadow", "1px 1px 1px rgba(255,255,255,0.8)")
        .text(displayName);
      
      // Add document count indicator
      if (d.data.documents && d.data.documents.length > 1) {
        labelGroup.append("text")
          .attr("dy", "1.8em")
          .attr("text-anchor", isRightSide ? "start" : "end")
          .style("font-size", "9px")
          .style("font-weight", "500")
          .style("fill", "#6b7280")
          .style("text-shadow", "1px 1px 1px rgba(255,255,255,0.8)")
          .text(`${d.data.documents.length} docs`);
      }
      
      // Add key terms indicator for clusters
      if (d.data.documents && d.data.documents.length > 1) {
        const topTerms = _.chain(d.data.documents)
          .flatMap(doc => Object.entries(doc.terms))
          .groupBy(([term]) => term)
          .mapValues(group => _.sumBy(group, ([, freq]) => freq))
          .toPairs()
          .orderBy(([, freq]) => freq, 'desc')
          .take(2)
          .map(([term]) => term)
          .value();
        
        if (topTerms.length > 0) {
          let termsText = topTerms.join(', ');
          if (termsText.length > 20) {
            termsText = termsText.substring(0, 17) + '...';
          }
          
          labelGroup.append("text")
            .attr("dy", "3.1em")
            .attr("text-anchor", isRightSide ? "start" : "end")
            .style("font-size", "8px")
            .style("font-weight", "400")
            .style("fill", "#9ca3af")
            .style("font-style", "italic")
            .style("text-shadow", "1px 1px 1px rgba(255,255,255,0.8)")
            .text(termsText);
        }
      }
      
      // Add visual connector line
      labelGroup.append("line")
        .attr("x1", isRightSide ? -15 : 15)
        .attr("y1", 0)
        .attr("x2", isRightSide ? -5 : 5)
        .attr("y2", 0)
        .style("stroke", "#d1d5db")
        .style("stroke-width", 1)
        .style("opacity", 0.6);
    });
  }, [documents]); // Add documents as dependency

  // File processing with BERTopic-inspired clustering
  const processFiles = async () => {
    if (selectedFiles.length === 0) return;

    setIsProcessing(true);
    const processedDocs = [];

    for (const file of selectedFiles) {
      try {
        let text = '';
        const extension = file.name.split('.').pop().toLowerCase();

        if (extension === 'txt') {
          text = await parseTextFile(file);
        } else if (extension === 'docx' || extension === 'doc') {
          text = await parseWordDoc(file);
        } else if (extension === 'pdf') {
          text = await parsePDF(file);
        } else {
          text = await parseTextFile(file);
        }

        if (text.length < 50) {
          console.warn(`Skipping ${file.name}: insufficient content`);
          continue;
        }

        const terms = extractTerms(text);
        if (Object.keys(terms).length === 0) {
          console.warn(`Skipping ${file.name}: no meaningful terms extracted`);
          continue;
        }

        processedDocs.push({
          name: file.name,
          text: text,
          terms,
          size: file.size
        });
      } catch (error) {
        console.error(`Error processing ${file.name}:`, error);
      }
    }

    setDocuments(processedDocs);
    
    if (processedDocs.length > 0) {
      const clusterResult = await bertopicClustering(processedDocs);
      setClustering(clusterResult);
    } else {
      setClustering(null);
    }
    
    setIsProcessing(false);
  };

  // Update visualization when clustering changes
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
    <div className="app-container">
      <div className="max-width-container">
        <div className="text-center mb-8">
          <h1 className="main-title">
            BERTopic Document Clustering
          </h1>
          <p className="subtitle">Advanced document clustering with automatic topic naming</p>
        </div>

        <div className="main-grid">
          {/* File Upload and Management */}
          <div className="card">
            <h2 className="section-title">Upload Documents</h2>
            
            <div className="mb-6">
              <input
                type="file"
                ref={fileInputRef}
                multiple
                accept=".txt,.pdf,.doc,.docx,.pages"
                onChange={handleFileSelect}
                className="file-input"
              />
              <div className="button-group">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="btn btn-primary"
                >
                  Select Files
                </button>
                <button
                  onClick={processFiles}
                  disabled={selectedFiles.length === 0 || isProcessing}
                  className="btn btn-secondary"
                >
                  {isProcessing ? (
                    <span className="flex items-center">
                      <div className="spinner"></div>
                      Processing...
                    </span>
                  ) : (
                    'Cluster Documents'
                  )}
                </button>
              </div>
            </div>

            {/* Selected Files List */}
            {selectedFiles.length > 0 && (
              <div className="mb-6">
                <h3 className="subsection-title">Selected Files ({selectedFiles.length}):</h3>
                <div className="file-list custom-scrollbar">
                  {selectedFiles.map((file, index) => (
                    <div key={index} className="file-item">
                      <div className="file-info">
                        <span className="file-name">{file.name}</span>
                        <div className="file-size">{(file.size / 1024).toFixed(1)} KB</div>
                      </div>
                      <button
                        onClick={() => removeFile(index)}
                        className="btn-remove"
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Processed Documents */}
            {documents.length > 0 && (
              <div>
                <h3 className="subsection-title">Processed Documents ({documents.length}):</h3>
                <div className="file-list documents custom-scrollbar">
                  {documents.map((doc, index) => (
                    <div key={index} className="document-card">
                      <div className="document-name">{doc.name}</div>
                      <div className="document-meta">
                        {Object.keys(doc.terms).length} unique terms
                      </div>
                      <div className="document-terms">
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

          {/* Visualization */}
          <div className="card">
            <h2 className="section-title">Topic Clusters</h2>
            
            {clustering ? (
              <div className="visualization-container">
                <svg ref={svgRef}></svg>
              </div>
            ) : (
              <div className="visualization-placeholder">
                <div className="placeholder-icon">🎯</div>
                <p className="placeholder-title">Upload documents to see topic clusters</p>
                <p className="placeholder-subtitle">Using BERTopic-inspired clustering with named topics</p>
              </div>
            )}
          </div>
        </div>

        {/* Enhanced Instructions */}
        <div className="instructions-card">
          <h3 className="instructions-title">How BERTopic Clustering Works:</h3>
          <div className="instructions-grid">
            <div>
              <h4>📄 Document Processing:</h4>
              <ul className="instruction-list">
                <li>• Extracts text from various file formats</li>
                <li>• Generates TF-IDF embeddings with n-grams</li>
                <li>• Preprocesses and normalizes content</li>
              </ul>
            </div>
            <div>
              <h4>🎯 Smart Clustering:</h4>
              <ul className="instruction-list">
                <li>• Uses DBSCAN clustering on embeddings</li>
                <li>• Automatically determines cluster count</li>
                <li>• Generates meaningful topic names using c-TF-IDF</li>
              </ul>
            </div>
          </div>
          
          <div className="tip-box">
            <p className="tip-text">
              <strong>💡 Tip:</strong> Upload documents with similar themes for better clustering results. 
              The algorithm works best with at least 3-5 documents covering related topics.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentClusteringApp;