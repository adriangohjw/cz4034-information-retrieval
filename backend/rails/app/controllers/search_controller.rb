class SearchController < ApplicationController

  def index
    @posts = Post.get_search_results(search_term: params[:search_term],
                                     hashtags: clean_array_params(params[:hashtags]))
  end

  private

  def clean_array_params(arr)
    return arr.blank? ? Array.new : arr
  end

end
