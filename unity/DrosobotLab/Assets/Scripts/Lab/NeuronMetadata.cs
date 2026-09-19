// Espelho de unity_assets/cns/neuron_metadata.json, escrito por
// blender/export_unity.py.
//
// Os campos carregam procedencia diferente e a interface precisa saber disso:
//
//   bodyId, type, side, neurotransmitter, group   DATA        (conectoma)
//   polarity                                      MODEL       (regra de Shiu et al.)
//   roleColor                                     apresentacao (escolha nossa)

using System;
using UnityEngine;

namespace Drosobot.Lab
{
    [Serializable]
    public class NeuronEntry
    {
        public long bodyId;
        public string objectName;
        public string group;
        public string side;
        public string type;
        public string instance;
        public string neurotransmitter;
        public string polarity;
        public float[] roleColor;
    }

    [Serializable]
    public class EdgeEntry
    {
        public long pre;        // DATA
        public long post;       // DATA
        public int weight;      // DATA -- contagem de sinapse EM, agregada por par
        public int sign;        // MODEL -- regra de Shiu et al. sobre o NT do pre
    }

    [Serializable]
    public class NeuronMetadata
    {
        public string source;
        public string generator;
        public string note;
        public string[] contextMeshes;
        public int neuronCount;
        public NeuronEntry[] neurons;
        public int edgeCount;
        public EdgeEntry[] edges;

        public static NeuronMetadata FromJson(string json)
        {
            // JsonUtility nao le o dicionario groupColors, entao ele fica de fora
            // desta classe de proposito -- a cor ja vem por neuronio em roleColor.
            return JsonUtility.FromJson<NeuronMetadata>(json);
        }
    }

    /// <summary>Procedencia de um campo mostrado na interface.</summary>
    public enum Provenance { Data, Model, Assumption, Unknown }

    public static class ProvenanceUtil
    {
        public static Provenance Parse(string s)
        {
            switch (s)
            {
                case "data": return Provenance.Data;
                case "model": return Provenance.Model;
                case "assumption": return Provenance.Assumption;
                default: return Provenance.Unknown;
            }
        }

        /// <summary>Cor da etiqueta. Distinguir isso na tela e requisito do projeto.</summary>
        public static Color Color(Provenance p)
        {
            switch (p)
            {
                case Provenance.Data:       return new Color(0.45f, 0.80f, 0.55f); // verde
                case Provenance.Model:      return new Color(0.45f, 0.65f, 0.95f); // azul
                case Provenance.Assumption: return new Color(0.95f, 0.70f, 0.30f); // ambar
                default:                    return new Color(0.6f, 0.6f, 0.6f);
            }
        }

        public static string Label(Provenance p)
        {
            switch (p)
            {
                case Provenance.Data:       return "DATA";
                case Provenance.Model:      return "MODEL";
                case Provenance.Assumption: return "ASSUMPTION";
                default:                    return "?";
            }
        }
    }
}
