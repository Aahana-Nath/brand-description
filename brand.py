import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import time
import os
from openai import OpenAI
from urllib.parse import urlparse
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="AI Brand Analyzer", page_icon="🔍", layout="wide")

class BrandAnalyzer:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_website_content(self, url):
        """Extract basic content from website with improved error handling"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Try HTTPS first, then HTTP if it fails
            urls_to_try = [url]
            if url.startswith('https://'):
                urls_to_try.append(url.replace('https://', 'http://'))
            
            # Enhanced headers to appear more like a real browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            for attempt_url in urls_to_try:
                try:
                    response = self.session.get(
                        attempt_url, 
                        timeout=15, 
                        headers=headers,
                        allow_redirects=True,
                        verify=False  # Skip SSL verification for problematic sites
                    )
                    
                    # Check if we got a successful response
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Remove scripts and styles
                        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                            element.decompose()
                        
                        # Get text
                        text = soup.get_text(separator=' ', strip=True)
                        text = ' '.join(text.split())[:3000]  # Limit text
                        
                        # Get title
                        title = soup.find('title')
                        title = title.get_text().strip() if title else ""
                        
                        # If we got meaningful content, return it
                        if len(text) > 100:  # Ensure we got substantial content
                            return {'title': title, 'content': text, 'url': attempt_url}
                        
                    elif response.status_code == 403:
                        continue  # Try next URL or fail
                    else:
                        continue  # Try next URL
                        
                except requests.exceptions.SSLError:
                    # SSL issues, try HTTP version
                    continue
                except requests.exceptions.ConnectionError:
                    # Connection issues, try next URL
                    continue
                except requests.exceptions.Timeout:
                    # Timeout, try next URL
                    continue
                except Exception as e:
                    # Other errors, try next URL
                    continue
            
            # If all attempts failed, try a different approach
            raise Exception(f"All connection attempts failed. Site may be blocking requests or URL is invalid.")
                
        except Exception as e:
            raise Exception(f"Website scraping failed: {str(e)}")
    
    def analyze_brand_fallback(self, url, brand_name=""):
        """Analyze brand using URL/domain when scraping fails"""
        try:
            # Extract domain name
            parsed = urlparse(url if url.startswith(('http://', 'https://')) else f'https://{url}')
            domain = parsed.netloc.replace('www.', '')
            
            # Create prompt based on URL/domain
            prompt = f"""
Analyze this brand based on their website URL and brand name.

Website: {url}
Domain: {domain}
Brand Name: {brand_name}

Based on the domain name and any context clues, create a professional brand description.

Create a JSON response with this structure:
{{
    "brand_name": "{brand_name or domain.split('.')[0].title()}",
    "brand_description": "A professional 50-100 word description of what this brand likely represents based on their domain name, industry context, and brand name. Focus on their probable visual identity and design approach.",
    "brand_personality": "Inferred brand personality in 15 words or less",
    "design_style": ["Professional", "Modern"],
    "target_audience": ["General consumers"],
    "tone": "Professional"
}}

Make educated guesses about their brand style based on:
- Industry suggested by domain name
- Brand name characteristics
- Common design patterns for similar businesses
"""
            
            # Call OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a brand analyst. Create professional brand descriptions based on available information. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {"status": "success", "data": result}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def analyze_brand(self, url, brand_name=""):
        """Analyze brand with fallback to URL-based analysis"""
        try:
            # First try to scrape the website
            website_data = self.get_website_content(url)
            
            # Create prompt for full analysis
            prompt = f"""
Analyze this website and create a brand description.

Website: {website_data['url']}
Title: {website_data['title']}
Content: {website_data['content']}

Create a JSON response with this structure:
{{
    "brand_name": "Brand Name",
    "brand_description": "A 50-100 word flowing paragraph describing the brand's visual style, aesthetic, design approach, and atmosphere. Write in complete sentences about their visual identity, color schemes, design philosophy, and overall brand feel.",
    "brand_personality": "Brief brand personality in 15 words or less",
    "design_style": ["Modern", "Clean", "Minimal"],
    "target_audience": ["Young adults", "Professionals"],
    "tone": "Friendly and Professional"
}}

Focus on visual and design elements. Write the brand_description as a natural paragraph, not bullet points.
"""
            
            # Call OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a brand analyst. Return only valid JSON with flowing brand descriptions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Ensure description exists
            if not result.get('brand_description', '').strip():
                result['brand_description'] = "This brand showcases a distinctive visual identity with thoughtful design choices that create a cohesive and engaging user experience."
            
            return {"status": "success", "data": result}
            
        except Exception as scraping_error:
            # If scraping fails, try fallback analysis
            st.warning(f"Website scraping failed for {url}, using domain-based analysis")
            return self.analyze_brand_fallback(url, brand_name)

def clean_dataframe(df):
    """Clean and prepare dataframe"""
    # Convert column names
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # Fill NaN with empty strings
    df = df.fillna('')
    
    # Convert to strings
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    
    return df

def find_columns(df):
    """Find relevant columns in dataframe"""
    brand_cols = [col for col in df.columns if 'brand' in col and 'name' in col]
    if not brand_cols:
        brand_cols = [col for col in df.columns if 'name' in col]
    
    url_cols = [col for col in df.columns if any(keyword in col for keyword in ['url', 'website', 'site'])]
    
    return brand_cols, url_cols

def get_value(row, columns):
    """Get first non-empty value from columns"""
    for col in columns:
        if col in row.index:
            value = str(row[col]).strip()
            if value and value.lower() not in ['nan', 'none', 'null', '']:
                return value
    return ""

def main():
    st.title("AI Brand Analyzer")
    st.write("Upload a CSV/Excel file or analyze individual brands")
    
    # API Key - Hardcoded
    api_key = "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"
    
    # Initialize analyzer
    analyzer = BrandAnalyzer(api_key)
    
    # Test API connection
    try:
        test = analyzer.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        st.success("API connection verified")
    except Exception as e:
        st.error(f"API connection failed: {e}")
        return
    
    tab1, tab2 = st.tabs(["File Upload", "Single Analysis"])
    
    with tab1:
        st.header("Batch Analysis")
        
        # Sample CSV download option
        st.subheader("📥 Download Sample CSV")
        sample_data = pd.DataFrame({
            'brand_name': ['Nike', 'Apple', 'Starbucks'],
            'website_url': ['https://www.nike.com', 'https://www.apple.com', 'https://www.starbucks.com']
        })
        sample_csv = sample_data.to_csv(index=False)
        st.download_button(
            label="Download Sample CSV Template",
            data=sample_csv,
            file_name="sample_brands.csv",
            mime="text/csv",
            help="Download a sample CSV file to see the expected format"
        )
        st.info("Your CSV should have columns for brand name and website URL")
        
        st.divider()
        
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
        
        if uploaded_file:
            try:
                # Read file with better encoding handling
                if uploaded_file.name.endswith('.csv'):
                    # Try different encodings for CSV files
                    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
                    df = None
                    
                    for encoding in encodings_to_try:
                        try:
                            uploaded_file.seek(0)  # Reset file pointer
                            df = pd.read_csv(uploaded_file, dtype=str, encoding=encoding)
                            st.info(f"Successfully read CSV with {encoding} encoding")
                            break
                        except UnicodeDecodeError:
                            continue
                        except pd.errors.EmptyDataError:
                            st.error("The CSV file is empty.")
                            return
                        except pd.errors.ParserError as e:
                            st.error(f"Error parsing CSV file: {str(e)}")
                            return
                        except Exception:
                            continue
                    
                    if df is None:
                        # Try reading as bytes and detecting encoding
                        try:
                            import chardet
                            raw_data = uploaded_file.read()
                            encoding_result = chardet.detect(raw_data)
                            detected_encoding = encoding_result['encoding']
                            uploaded_file.seek(0)  # Reset file pointer
                            df = pd.read_csv(uploaded_file, dtype=str, encoding=detected_encoding)
                            st.info(f"Successfully read CSV with detected encoding: {detected_encoding}")
                        except:
                            uploaded_file.seek(0)  # Reset file pointer
                            df = pd.read_csv(uploaded_file, dtype=str, encoding='latin-1', errors='ignore')
                            st.warning("Used fallback encoding (latin-1) - some characters may be incorrect")
                else:
                    try:
                        df = pd.read_excel(uploaded_file, dtype=str)
                    except pd.errors.EmptyDataError:
                        st.error("The Excel file is empty.")
                        return
                    except Exception as e:
                        st.error(f"Error reading Excel file: {str(e)}")
                        return
                
                df = clean_dataframe(df)
                
                # Validate dataframe
                if df.empty:
                    st.error("The file contains no data rows.")
                    return
                
                if len(df.columns) == 0:
                    st.error("The file has no columns. Please check your file format.")
                    return
                
                # Check for meaningful data
                if len(df.dropna(how='all')) == 0:
                    st.error("The file contains only empty rows.")
                    return
                
                st.write(f"Loaded {len(df)} rows and {len(df.columns)} columns")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Find columns
                brand_cols, url_cols = find_columns(df)
                
                if not brand_cols:
                    st.error("No brand name column found")
                    return
                if not url_cols:
                    st.error("No URL column found")
                    return
                
                st.info(f"Brand column: {brand_cols[0]}")
                st.info(f"URL column: {url_cols[0]}")
                
                # Analysis settings
                col1, col2 = st.columns(2)
                with col1:
                    num_to_process = st.number_input("Number of brands to analyze", 1, len(df), len(df))
                with col2:
                    delay = st.number_input("Delay between requests (seconds)", 1, 10, 2)
                
                if st.button("Start Analysis", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Add result columns
                    df['brand_description'] = ''
                    df['brand_personality'] = ''
                    df['design_style'] = ''
                    df['target_audience'] = ''
                    df['tone'] = ''
                    df['analysis_status'] = ''
                    
                    success_count = 0
                    
                    for i in range(min(num_to_process, len(df))):
                        row = df.iloc[i]
                        
                        # Get data
                        brand_name = get_value(row, brand_cols) or f"Brand_{i+1}"
                        url = get_value(row, url_cols)
                        
                        status_text.text(f"Processing {brand_name} ({i+1}/{num_to_process})")
                        
                        if url:
                            try:
                                result = analyzer.analyze_brand(url, brand_name)
                                
                                if result['status'] == 'success':
                                    data = result['data']
                                    
                                    df.at[i, 'brand_description'] = data.get('brand_description', 'Analysis completed')
                                    df.at[i, 'brand_personality'] = data.get('brand_personality', '')
                                    df.at[i, 'design_style'] = ', '.join(data.get('design_style', []))
                                    df.at[i, 'target_audience'] = ', '.join(data.get('target_audience', []))
                                    df.at[i, 'tone'] = data.get('tone', '')
                                    df.at[i, 'analysis_status'] = 'Success'
                                    success_count += 1
                                    
                                    preview = data.get('brand_description', '')[:60] + "..." if len(data.get('brand_description', '')) > 60 else data.get('brand_description', '')
                                    status_text.text(f"✅ {brand_name}: {preview}")
                                    
                                else:
                                    df.at[i, 'brand_description'] = f"Error: {result['error']}"
                                    df.at[i, 'analysis_status'] = 'Failed'
                                    
                            except Exception as e:
                                df.at[i, 'brand_description'] = f"Exception: {str(e)}"
                                df.at[i, 'analysis_status'] = 'Exception'
                        else:
                            df.at[i, 'brand_description'] = "No URL found"
                            df.at[i, 'analysis_status'] = 'No URL'
                        
                        progress_bar.progress((i + 1) / num_to_process)
                        time.sleep(delay)
                    
                    st.success(f"Analysis complete! {success_count}/{num_to_process} successful")
                    
                    # Show results
                    st.subheader("Results")
                    result_df = df.head(num_to_process)
                    st.dataframe(result_df)
                    
                    # Download
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        f"brand_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                    
            except Exception as e:
                st.error(f"File processing error: {e}")
    
    with tab2:
        st.header("Single Brand Analysis")
        
        with st.form("single_analysis"):
            brand_name = st.text_input("Brand Name")
            website_url = st.text_input("Website URL")
            submit = st.form_submit_button("Analyze")
        
        if submit and website_url:
            with st.spinner(f"Analyzing {brand_name or 'brand'}..."):
                result = analyzer.analyze_brand(website_url, brand_name)
                
                if result['status'] == 'success':
                    data = result['data']
                    
                    st.success("Analysis Complete!")
                    st.header(data.get('brand_name', brand_name))
                    
                    # Brand description
                    st.subheader("Brand Description")
                    st.write(data['brand_description'])
                    
                    # Other details
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Personality")
                        st.write(data.get('brand_personality', 'N/A'))
                        
                        st.subheader("Tone")
                        st.write(data.get('tone', 'N/A'))
                    
                    with col2:
                        st.subheader("Design Style")
                        styles = data.get('design_style', [])
                        if styles:
                            for style in styles:
                                st.write(f"• {style}")
                        
                        st.subheader("Target Audience")
                        audience = data.get('target_audience', [])
                        if audience:
                            for aud in audience:
                                st.write(f"• {aud}")
                    
                    # Download single result
                    single_result = pd.DataFrame([{
                        'brand_name': data.get('brand_name', brand_name),
                        'website_url': website_url,
                        'brand_description': data['brand_description'],
                        'brand_personality': data.get('brand_personality', ''),
                        'design_style': ', '.join(data.get('design_style', [])),
                        'target_audience': ', '.join(data.get('target_audience', [])),
                        'tone': data.get('tone', '')
                    }])
                    
                    csv = single_result.to_csv(index=False)
                    st.download_button(
                        "Download Analysis",
                        csv,
                        f"{(brand_name or 'brand').replace(' ', '_')}_analysis.csv",
                        "text/csv"
                    )
                else:
                    st.error(f"Analysis failed: {result['error']}")

if __name__ == "__main__":
    main()