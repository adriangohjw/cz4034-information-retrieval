class CreatePosts < ActiveRecord::Migration[5.2]
  def change
    create_table :posts do |t|
      t.string :unique_id
      t.string :body
      t.json :hashtags
      t.string :creator
      t.boolean :verified
      t.integer :followers
      t.integer :following
      t.integer :impressions
      t.integer :upvotes
      t.integer :reposts
      t.datetime :posted_at

      t.timestamps
    end
  end
end
