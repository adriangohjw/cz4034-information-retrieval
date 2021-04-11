class AddScoresToPost < ActiveRecord::Migration[5.2]
  def change
    add_column :posts, :creator_score, :float
    add_column :posts, :reach_score, :float
  end
end
