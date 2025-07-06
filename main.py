import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import asyncio
import time
from docx import Document
from fpdf import FPDF
from PIL import ImageTk, Image
import os

# --- Global Variables for Lazy Loading ---
plag_check_module = None
models_loaded = False

def load_models_async():
    """Load models in background thread with thread-safe GUI updates - FIXED"""
    global plag_check_module, models_loaded
    
    def update_status(message, progress_value):
        """Thread-safe GUI update - FIXED"""
        def update_gui():
            try:
                status_var.set(message)
                progress['value'] = progress_value
                root.update_idletasks()
            except:
                pass  # Ignore if widgets are destroyed
        try:
            root.after(0, update_gui)
        except:
            pass  # Ignore if main thread is closing
    
    try:
        update_status("Initializing AI models (this may take 1-2 minutes)...", 10)
        
        # Add current directory to Python path
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        update_status("Loading sentence transformer model...", 30)
        
        # Import with better error handling
        try:
            from plag_check import (
                extract_text_from_docx,
                extract_text_from_pdf,
                analyze_sentences_async,
                analyze_ai_only,
                generate_graphs
            )
        except ImportError as ie:
            update_status(f"Import error: {str(ie)}", 0)
            return
        
        update_status("Loading GPT-2 model...", 60)
        
        # Give models time to fully load
        time.sleep(2)
        
        plag_check_module = {
            'extract_text_from_docx': extract_text_from_docx,
            'extract_text_from_pdf': extract_text_from_pdf,
            'analyze_sentences_async': analyze_sentences_async,
            'analyze_ai_only': analyze_ai_only,
            'generate_graphs': generate_graphs
        }
        
        update_status("✅ Models loaded successfully - Ready for analysis!", 100)
        models_loaded = True
        
        # Reset progress bar after 2 seconds
        def reset_progress():
            try:
                progress['value'] = 0
            except:
                pass  # Ignore if widget is destroyed
        try:
            root.after(2000, reset_progress)
        except:
            pass  # Ignore if main thread is closing
        
    except Exception as e:
        update_status(f"❌ Error loading models: {str(e)}", 0)
        models_loaded = False
        print(f"Detailed error: {e}")  # For debugging

# --- Setup ---
graph_img = None
graph_label = None

root = tk.Tk()
root.title("Debnil's Open Source Plagiarism & AI Detector v10.4")
root.geometry("1600x900")
root.configure(bg="#f0f0f0")

# --- Main Layout: Left side for content, Right side for graphs ---
main_frame = tk.Frame(root, bg="#f0f0f0")
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Left side - Content area
left_frame = tk.Frame(main_frame, bg="#f0f0f0", width=800)
left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

# Right side - Graph area (expanded to fill remaining space)
right_frame = tk.Frame(main_frame, bg="#e8e8e8", width=700)
right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))
right_frame.pack_propagate(False)  # Maintain fixed width

# --- Status Label ---
status_var = tk.StringVar()
status_var.set("Debnil - built from scratch, open for all")
status_label = tk.Label(left_frame, textvariable=status_var, bg="#f0f0f0", font=("Arial", 10, "italic"))
status_label.pack()

# --- Progress Bar ---
progress = ttk.Progressbar(left_frame, orient=tk.HORIZONTAL, length=500, mode='determinate')
progress.pack(pady=(0, 10))

# --- Control Panel ---
top_frame = tk.Frame(left_frame, bg="#f0f0f0")
top_frame.pack(pady=10)

# --- Text Box ---
text_box = tk.Text(left_frame, wrap="word", font=("Arial", 12), bg="#FFFFFF")
text_box.pack(expand=True, fill="both", padx=0, pady=(0, 10))

# --- Enhanced Graph Area Setup ---
# Graph control panel
graph_control_frame = tk.Frame(right_frame, bg="#e8e8e8")
graph_control_frame.pack(fill="x", padx=10, pady=5)

graph_title = tk.Label(graph_control_frame, text="📊 Analysis Graphs", font=("Arial", 14, "bold"), bg="#e8e8e8")
graph_title.pack(side="left")

# Graph type selection dropdowns
graph1_var = tk.StringVar(value="pie_chart")
graph2_var = tk.StringVar(value="ai_trend")

graph1_label = tk.Label(graph_control_frame, text="Top Graph:", font=("Arial", 10), bg="#e8e8e8")
graph1_label.pack(side="left", padx=(20, 5))

graph1_combo = ttk.Combobox(graph_control_frame, textvariable=graph1_var, width=12, state="readonly")
graph1_combo['values'] = ("pie_chart", "ai_trend", "plagiarism_trend", "source_freq", "radar_chart")
graph1_combo.pack(side="left", padx=(0, 10))

graph2_label = tk.Label(graph_control_frame, text="Bottom Graph:", font=("Arial", 10), bg="#e8e8e8")
graph2_label.pack(side="left", padx=(10, 5))

graph2_combo = ttk.Combobox(graph_control_frame, textvariable=graph2_var, width=12, state="readonly")
graph2_combo['values'] = ("ai_trend", "plagiarism_trend", "source_freq", "stacked_bar", "radar_chart")
graph2_combo.pack(side="left", padx=(0, 10))

# Update button
update_graphs_btn = tk.Button(graph_control_frame, text="🔄 Update", command=lambda: update_graph_display(), 
                             bg="#17a2b8", fg="white", font=("Arial", 8, "bold"))
update_graphs_btn.pack(side="right", padx=(10, 0))

# Upper graph frame (expanded)
upper_graph_frame = tk.Frame(right_frame, bg="#e8e8e8", height=380)
upper_graph_frame.pack(fill="both", expand=True, padx=10, pady=5)
upper_graph_frame.pack_propagate(False)

upper_graph_label = tk.Label(upper_graph_frame, text="Pie Chart - Content Breakdown", font=("Arial", 10, "bold"), bg="#e8e8e8")
upper_graph_label.pack()

upper_graph_canvas = tk.Label(upper_graph_frame, bg="#ffffff", text="Graph will appear here after analysis", fg="#666666")
upper_graph_canvas.pack(fill="both", expand=True, padx=5, pady=5)

# Lower graph frame (expanded)
lower_graph_frame = tk.Frame(right_frame, bg="#e8e8e8", height=380)
lower_graph_frame.pack(fill="both", expand=True, padx=10, pady=5)
lower_graph_frame.pack_propagate(False)

lower_graph_label = tk.Label(lower_graph_frame, text="AI Detection Trend", font=("Arial", 10, "bold"), bg="#e8e8e8")
lower_graph_label.pack()

lower_graph_canvas = tk.Label(lower_graph_frame, bg="#ffffff", text="Graph will appear here after analysis", fg="#666666")
lower_graph_canvas.pack(fill="both", expand=True, padx=5, pady=5)

# --- Global Variables ---
current_text = ""
last_analysis_results = None
last_analysis_type = None

# --- Helper Functions ---
def check_models_loaded():
    """Check if models are loaded before proceeding"""
    if not models_loaded:
        messagebox.showwarning("Models Loading", "AI models are still loading. Please wait a moment and try again.")
        return False
    return True

def display_graph(graph_path, canvas_widget):
    """Display graph in the specified canvas with better sizing"""
    try:
        if os.path.exists(graph_path):
            # Open and resize image to fit canvas better
            img = Image.open(graph_path)
            # Get canvas dimensions
            canvas_width = canvas_widget.winfo_width()
            canvas_height = canvas_widget.winfo_height()
            
            # Use larger default size if canvas isn't sized yet
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 650, 350
            
            # Resize maintaining aspect ratio
            img_ratio = img.width / img.height
            canvas_ratio = canvas_width / canvas_height
            
            if img_ratio > canvas_ratio:
                new_width = canvas_width - 20
                new_height = int(new_width / img_ratio)
            else:
                new_height = canvas_height - 20
                new_width = int(new_height * img_ratio)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            canvas_widget.configure(image=photo, text="")
            canvas_widget.image = photo  # Keep a reference
        else:
            canvas_widget.configure(text="Graph not available", image="")
    except Exception as e:
        canvas_widget.configure(text=f"Error loading graph: {str(e)}", image="")

def update_graph_display():
    """Update graph display based on dropdown selections"""
    global last_analysis_results, last_analysis_type
    
    if not last_analysis_results:
        messagebox.showinfo("No Data", "Run an analysis first to generate graphs.")
        return
    
    try:
        # Update labels
        graph_names = {
            "pie_chart": "Pie Chart - Content Breakdown",
            "ai_trend": "AI Detection Trend",
            "plagiarism_trend": "Plagiarism Detection Trend", 
            "source_freq": "Source Frequency Distribution",
            "stacked_bar": "Stacked Analysis by Section",
            "radar_chart": "Multi-Metric Radar Chart"
        }
        
        upper_graph_label.config(text=graph_names.get(graph1_var.get(), "Unknown Graph"))
        lower_graph_label.config(text=graph_names.get(graph2_var.get(), "Unknown Graph"))
        
        # Generate new graphs
        if last_analysis_type == "full":
            ai_results = [(r[0], r[5], r[6]) for r in last_analysis_results]
            plag_results = [(r[0], r[1], r[2]) for r in last_analysis_results]  # FIXED: correct indices
            sources = [r[3] for r in last_analysis_results]
        else:  # ai_only
            ai_results = last_analysis_results
            plag_results = None
            sources = None
        
        graph_paths = plag_check_module['generate_graphs'](
            ai_results, plag_results, sources, 
            mode=last_analysis_type,
            graph1_type=graph1_var.get(),
            graph2_type=graph2_var.get()
        )
        
        if graph_paths:
            if "graph1" in graph_paths:
                display_graph(graph_paths["graph1"], upper_graph_canvas)
            if "graph2" in graph_paths:
                display_graph(graph_paths["graph2"], lower_graph_canvas)
    
    except Exception as e:
        messagebox.showerror("Graph Error", f"Error updating graphs: {str(e)}")

# --- Analysis Functions ---
def clean_text(text):
    """Clean and prepare text for analysis"""
    import re
    # Remove extra whitespace and split into sentences
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = re.split(r'[.!?]+', text)
    # Filter out very short sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences

def run_analysis_async(analysis_type="full"):
    """Run analysis in background thread - FIXED threading issues"""
    if not check_models_loaded():
        return
    
    global current_text, last_analysis_results, last_analysis_type
    
    current_text = text_box.get("1.0", tk.END).strip()
    if not current_text:
        messagebox.showwarning("No Text", "Please enter some text to analyze.")
        return
    
    # Clean and prepare text
    sentences = clean_text(current_text)
    if not sentences:
        messagebox.showwarning("No Valid Sentences", "Could not find valid sentences to analyze.")
        return
    
    # Update status
    status_var.set(f"Analyzing {len(sentences)} sentences...")
    progress['value'] = 0
    
    def analysis_thread():
        start_time = time.time()  # Track analysis time
        try:
            # Create new event loop for this thread
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run analysis
            if analysis_type == "full":
                results = loop.run_until_complete(plag_check_module['analyze_sentences_async'](sentences))
            else:  # ai_only
                results = loop.run_until_complete(plag_check_module['analyze_ai_only'](sentences))
            
            # Don't close the loop here - let it be garbage collected
            
            # Update results
            last_analysis_results = results
            last_analysis_type = analysis_type
            
            # Calculate time taken
            analysis_time = int(time.time() - start_time)
            
            # Update GUI in main thread - FIXED
            def update_results():
                try:
                    display_results(results, analysis_type, analysis_time)
                    generate_and_display_graphs(results, analysis_type)
                    status_var.set("✅ Analysis complete!")
                    progress['value'] = 100
                    
                    # Reset progress bar after 2 seconds
                    def reset_progress():
                        try:
                            progress['value'] = 0
                        except:
                            pass  # Ignore if widget is destroyed
                    root.after(2000, reset_progress)
                except Exception as e:
                    print(f"Error updating results: {e}")
            
            # Use try-except for thread-safe GUI updates
            try:
                root.after(0, update_results)
            except:
                pass  # Ignore if main thread is closing
            
        except Exception as e:
            def show_error():
                try:
                    messagebox.showerror("Analysis Error", f"Error during analysis: {str(e)}")
                    status_var.set("❌ Analysis failed")
                    progress['value'] = 0
                except:
                    pass  # Ignore if widget is destroyed
            try:
                root.after(0, show_error)
            except:
                pass  # Ignore if main thread is closing
    
    # Start analysis in background thread
    thread = threading.Thread(target=analysis_thread, daemon=True)
    thread.start()

def display_results(results, analysis_type, analysis_time):
    """Display analysis results in text box with summary"""
    text_box.delete("1.0", tk.END)
    
    if analysis_type == "full":
        # Full analysis summary
        total = len(results)
        ai_count = sum(1 for r in results if "AI" in r[5])  # r[5] is ai_verdict
        plag_count = sum(1 for r in results if "Copied" in r[2])  # r[2] is plag_verdict
        clean_count = total - ai_count - plag_count
        
        summary = f"""🔍 Full Analysis Summary
Total Sentences: {total}
AI Content: {round(ai_count / total * 100, 2)}% ({ai_count}/{total})
Plagiarized: {round(plag_count / total * 100, 2)}% ({plag_count}/{total})
Original: {round(clean_count / total * 100, 2)}% ({clean_count}/{total})
Time Taken: {analysis_time} sec

"""
        
        text_box.insert(tk.END, summary)
        text_box.insert(tk.END, "📊 DETAILED ANALYSIS RESULTS\n" + "="*50 + "\n\n")
        
        for i, (sentence, sim_score, verdict, source, _, ai_verdict, ai_score) in enumerate(results, 1):
            text_box.insert(tk.END, f"[{i}] {sentence[:100]}{'...' if len(sentence) > 100 else ''}\n")
            text_box.insert(tk.END, f"    🔍 Plagiarism: {sim_score}% - {verdict} (Source: {source})\n")
            text_box.insert(tk.END, f"    🤖 AI Detection: {ai_verdict} (Perplexity: {ai_score})\n\n")
    
    else:  # ai_only
        # AI-only analysis summary
        total = len(results)
        ai_count = sum(1 for r in results if "AI" in r[1])  # r[1] is verdict
        human_count = total - ai_count
        
        summary = f"""🧠 GPT-2 AI Detection Only Summary
AI Content: {round(ai_count / total * 100, 2)}% ({ai_count}/{total})
Human-like: {round(human_count / total * 100, 2)}% ({human_count}/{total})
Time Taken: {analysis_time} sec

"""
        
        text_box.insert(tk.END, summary)
        text_box.insert(tk.END, "🤖 DETAILED AI DETECTION RESULTS\n" + "="*50 + "\n\n")
        
        for i, (sentence, verdict, score) in enumerate(results, 1):
            text_box.insert(tk.END, f"[{i}] {sentence[:100]}{'...' if len(sentence) > 100 else ''}\n")
            text_box.insert(tk.END, f"    🤖 AI Detection: {verdict} (Perplexity: {score})\n\n")

def generate_and_display_graphs(results, analysis_type):
    """Generate and display graphs"""
    try:
        if analysis_type == "full":
            ai_results = [(r[0], r[5], r[6]) for r in results]
            plag_results = [(r[0], r[1], r[2]) for r in results]
            sources = [r[3] for r in results]
        else:  # ai_only
            ai_results = results
            plag_results = None
            sources = None
        
        graph_paths = plag_check_module['generate_graphs'](
            ai_results, plag_results, sources, 
            mode=analysis_type,
            graph1_type=graph1_var.get(),
            graph2_type=graph2_var.get()
        )
        
        if graph_paths:
            if "graph1" in graph_paths:
                display_graph(graph_paths["graph1"], upper_graph_canvas)
            if "graph2" in graph_paths:
                display_graph(graph_paths["graph2"], lower_graph_canvas)
    
    except Exception as e:
        print(f"Error generating graphs: {e}")

# --- File Operations ---
def load_file():
    """Load text from file"""
    file_path = filedialog.askopenfilename(
        title="Select File",
        filetypes=[
            ("Text files", "*.txt"),
            ("Word documents", "*.docx"),
            ("PDF files", "*.pdf"),
            ("All files", "*.*")
        ]
    )
    
    if file_path:
        try:
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_path.endswith('.docx'):
                if check_models_loaded():
                    content = plag_check_module['extract_text_from_docx'](file_path)
                else:
                    return
            elif file_path.endswith('.pdf'):
                if check_models_loaded():
                    content = plag_check_module['extract_text_from_pdf'](file_path)
                else:
                    return
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            text_box.delete("1.0", tk.END)
            text_box.insert("1.0", content)
            status_var.set(f"✅ Loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("File Error", f"Error loading file: {str(e)}")

def save_results():
    """Save analysis results to file"""
    if not last_analysis_results:
        messagebox.showwarning("No Results", "No analysis results to save.")
        return
    
    file_path = filedialog.asksaveasfilename(
        title="Save Results",
        defaultextension=".txt",
        filetypes=[
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
    )
    
    if file_path:
        try:
            content = text_box.get("1.0", tk.END)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            status_var.set(f"✅ Saved: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving file: {str(e)}")

def clear_text():
    """Clear text box"""
    text_box.delete("1.0", tk.END)
    status_var.set("Text cleared")

# --- Control Buttons ---
button_frame = tk.Frame(top_frame, bg="#f0f0f0")
button_frame.pack(fill="x", pady=5)

# File operations
load_btn = tk.Button(button_frame, text="📁 Load File", command=load_file, 
                    bg="#28a745", fg="white", font=("Arial", 10, "bold"))
load_btn.pack(side="left", padx=5)

save_btn = tk.Button(button_frame, text="💾 Save Results", command=save_results, 
                    bg="#6c757d", fg="white", font=("Arial", 10, "bold"))
save_btn.pack(side="left", padx=5)

clear_btn = tk.Button(button_frame, text="🗑️ Clear", command=clear_text, 
                     bg="#dc3545", fg="white", font=("Arial", 10, "bold"))
clear_btn.pack(side="left", padx=5)

# Analysis buttons
analyze_full_btn = tk.Button(button_frame, text="🔍 Full Analysis", 
                            command=lambda: run_analysis_async("full"), 
                            bg="#007bff", fg="white", font=("Arial", 10, "bold"))
analyze_full_btn.pack(side="right", padx=5)

analyze_ai_btn = tk.Button(button_frame, text="🤖 AI Only", 
                          command=lambda: run_analysis_async("ai_only"), 
                          bg="#17a2b8", fg="white", font=("Arial", 10, "bold"))
analyze_ai_btn.pack(side="right", padx=5)

# --- Initialize Models ---
def initialize_app():
    """Initialize the application"""
    # Start loading models in background
    threading.Thread(target=load_models_async, daemon=True).start()
    
    # Show initial message
    text_box.insert("1.0", "Welcome to Debnil's Plagiarism & AI Detector v10.4\n\n")
    text_box.insert(tk.END, "📝 Enter your text here or load a file to begin analysis.\n\n")
    text_box.insert(tk.END, "Features:\n")
    text_box.insert(tk.END, "• Full Analysis: Detects both plagiarism and AI-generated content\n")
    text_box.insert(tk.END, "• AI Only: Fast AI detection without plagiarism checking\n")
    text_box.insert(tk.END, "• Multiple graph types for visualization\n")
    text_box.insert(tk.END, "• Support for TXT, DOCX, and PDF files\n\n")
    text_box.insert(tk.END, "⚠️ Note: Models are loading in the background. Please wait for the ready message. Kindly note this is an open source model, use wisely. DO NOT OVER USE THE PLAGIARISM CHECKEING FUNCTION.\n")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        initialize_app()
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        # Ensure proper cleanup
        try:
            root.quit()
        except:
            pass